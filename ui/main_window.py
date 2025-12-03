from PyQt5.QtCore import Qt, pyqtSignal, QObject
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QSplitter,
    QVBoxLayout,
    QTabWidget,
    QFileDialog,
    QMessageBox,
)

from .workflow_tab import WorkflowTab
from .logic_tab import LogicTab
from .image_preview_panel import ImagePreviewPanel
from .defect_ledger import DefectLedger
from .qt_image import np_bgr_to_qpixmap
from services import project_loader
from services import camera_service
from services import camera_manager as cammgr
from services import turntable_service
from services import linear_axis_service
from services.config import settings, state, save_state
from services import solvision_manager
from services.app_paths import app_root
import concurrent.futures
import os
import cv2

# Horizontal FOV of the front camera when measured inside the top-camera image (pixels)
DEFAULT_FRONT_FOV_TOP_PX = 951.0


class _AxisUiBridge(QObject):
    set_ready = pyqtSignal(bool)
    set_calibrating = pyqtSignal(bool)
    set_calibrated = pyqtSignal(bool, object)  # object for Optional[float]
    set_position = pyqtSignal(object)

class MainWindow(QMainWindow):
    tt_message = pyqtSignal(str)
    tt_status = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Detectron Demo")
        self.resize(1184, 661)
        self._current_image_path = None

        # Root splitter (left: workflow tabs, right: previews + ledger)
        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)

        # Left side: Tabs
        self.tabs = QTabWidget()
        self.workflow_tab = WorkflowTab()
        self.tabs.addTab(self.workflow_tab, "Workflow")
        # Placeholder second tab matching C# dynamic UI intent
        self.logic_tab = LogicTab()
        self.tabs.addTab(self.logic_tab, "Logic")
        root_splitter.addWidget(self.tabs)

        # Right side: previews + ledger
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_panel = ImagePreviewPanel()
        right_layout.addWidget(self.preview_panel)
        try:
            st = state()
            if st.overlay_enabled is not None:
                self.preview_panel.chk_overlay.setChecked(bool(st.overlay_enabled))
        except Exception:
            pass

        self.defect_ledger = DefectLedger()
        right_layout.addWidget(self.defect_ledger)

        root_splitter.addWidget(right_container)
        root_splitter.setStretchFactor(0, 38)
        root_splitter.setStretchFactor(1, 62)

        self.setCentralWidget(root_splitter)

        # Route detectron debug logs to UI Log as well (when DETECTRON_DEBUG=1)
        try:
            solvision_manager.set_ui_logger(lambda line: self.tt_message.emit(line))
        except Exception:
            pass

        # Wire signals
        self.workflow_tab.load_image_requested.connect(self.on_load_image)
        self.workflow_tab.run_detection_requested.connect(self.on_run_detection)
        self.workflow_tab.open_tuner_requested.connect(self.on_open_tuner)
        self.workflow_tab.load_attachment_requested.connect(self.on_load_attachment_file)
        self.workflow_tab.load_front_requested.connect(self.on_load_front_file)
        self.workflow_tab.load_defect_requested.connect(self.on_load_defect_file)
        self.workflow_tab.run_step3_step4_requested.connect(self.on_run_step3_step4_existing)
        try:
            self.workflow_tab.defect_threshold_changed.connect(self.on_defect_threshold_changed)
        except Exception:
            pass
        

        self.preview_panel.overlay_toggled.connect(self.on_overlay_toggled)
        self.preview_panel.prev_requested.connect(self.on_prev)
        self.preview_panel.next_requested.connect(self.on_next)

        # Camera panel signals
        cam = self.workflow_tab.camera_panel
        cam.refresh_requested.connect(self.on_camera_refresh)
        cam.connect_requested.connect(self.on_camera_connect)
        cam.disconnect_requested.connect(self.on_camera_disconnect)
        cam.capture_requested.connect(self.on_camera_capture)
        cam.selection_changed.connect(self.on_camera_selected)

        # Initial device list
        self.on_camera_refresh()

        # No backend selection to restore (Detectron only)

        # Turntable panel signals
        tt = self.workflow_tab.turntable_panel
        tt.refresh_requested.connect(self.on_turntable_refresh)
        tt.connect_requested.connect(self.on_turntable_connect)
        tt.disconnect_requested.connect(self.on_turntable_disconnect)
        tt.home_requested.connect(self.on_turntable_home)
        tt.rotate_requested.connect(self.on_turntable_rotate)
        tt.port_selected.connect(self.on_turntable_port_selected)
        tt.step_changed.connect(self.on_turntable_step_changed)
        # Subscribe to turntable messages for logging (thread-safe via signal relay)
        self.tt_message.connect(self._handle_turntable_message)
        self.tt_status.connect(self._handle_turntable_status)
        self._tt_listener = self._on_tt_raw_message
        turntable_service.add_listener(self._tt_listener)
        self.on_turntable_refresh()

        # Linear axis panel signals
        ax = self.workflow_tab.linear_axis_panel
        ax.refresh_requested.connect(self.on_axis_refresh)
        ax.connect_requested.connect(self.on_axis_connect)
        ax.disconnect_requested.connect(self.on_axis_disconnect)
        ax.calibrate_requested.connect(self.on_axis_calibrate)
        ax.home_requested.connect(self.on_axis_home)
        ax.goto_requested.connect(self.on_axis_goto)
        ax.home_set_requested.connect(self.on_axis_home_set)
        # Thread-safe axis UI bridge
        self._axis_ui = _AxisUiBridge()
        self._axis_ui.set_ready.connect(ax.set_ready)
        self._axis_ui.set_calibrating.connect(ax.set_calibrating)
        self._axis_ui.set_calibrated.connect(lambda ok, p=None: ax.set_calibrated(ok, p if p is not None else None))
        self._axis_ui.set_position.connect(lambda p: ax.set_position(p))
        # Initial axis ports
        self.on_axis_refresh()

        # Reflect existing connections from wizard
        try:
            top_idx = camera_service.get_connected_index("Top")
            if top_idx is not None:
                self.workflow_tab.camera_panel.set_connected("Top", True, "")
            front_idx = camera_service.get_connected_index("Front")
            if front_idx is not None:
                self.workflow_tab.camera_panel.set_connected("Front", True, "")
            # Restore saved selections
            st = state()
            if top_idx is None and st.camera_top_index is not None:
                self.workflow_tab.camera_panel.set_selected_index("Top", int(st.camera_top_index))
            if front_idx is None and st.camera_front_index is not None:
                self.workflow_tab.camera_panel.set_selected_index("Front", int(st.camera_front_index))
            # No single model box; Selected Files reflects persisted paths
        except Exception:
            pass
        try:
            if turntable_service.is_connected():
                self.workflow_tab.turntable_panel.set_connected(True, turntable_service.port_name())
            else:
                st = state()
                if st.turntable_port:
                    idx = self.workflow_tab.turntable_panel.port_combo.findText(st.turntable_port)
                    if idx >= 0:
                        self.workflow_tab.turntable_panel.port_combo.setCurrentIndex(idx)
                if st.turntable_step is not None:
                    self.workflow_tab.turntable_panel.step.setValue(float(st.turntable_step))
        except Exception:
            pass

        # Seed previews from wizard if available
        try:
            s = settings()
            # Set Selected Files group from saved state
            st = state()
            try:
                self.workflow_tab.set_selected_files(st.attachment_path, st.front_attachment_path, st.defect_path)
            except Exception:
                pass

            if s.top_preview_np is not None:
                pm = np_bgr_to_qpixmap(s.top_preview_np)
                self.preview_panel.set_original_np(pm)
            if s.front_preview_np is not None:
                pm = np_bgr_to_qpixmap(s.front_preview_np)
                self.preview_panel.set_front_np(pm)
        except Exception:
            pass

    # Slots
    def on_load_project(self):
        # Legacy handler unused (single model box removed). Use Selected Files instead.
        pass

    def on_load_selected_project(self, path: str):
        # Legacy handler unused (single model box removed). Use Selected Files instead.
        pass

    def on_load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)")
        if not path:
            return
        self._current_image_path = path
        try:
            st = state(); st.last_image_path = path; save_state()
        except Exception:
            pass
        self.workflow_tab.append_log(f"Loaded image: {path}")
        self.preview_panel.set_original_image(path)
        self.preview_panel.set_front_preview_image(path)

    def on_run_detection(self):
        # Prepare capture directory structure based on date/time
        from datetime import datetime
        import re
        part_id_raw = ""
        try:
            part_id_raw = self.workflow_tab.part_id()
        except Exception:
            part_id_raw = ""
        if not part_id_raw:
            QMessageBox.information(self, "Run Detection", "Please enter a Part ID before running detection.")
            return
        part_id_clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part_id_raw).strip("_")
        if not part_id_clean:
            part_id_clean = "part"
        try:
            st = state(); st.part_id = part_id_raw; save_state()
        except Exception:
            pass
        # Start cycle timer at button press (covers detection + motions)
        try:
            self._cycle_start_ts = time.time()
        except Exception:
            self._cycle_start_ts = None
        try:
            self.workflow_tab.append_log(f"[Capture] Part ID set to '{part_id_raw}' (folder '{part_id_clean}')")
        except Exception:
            pass
        cap_dir = None
        try:
            ts = datetime.now()
            base = app_root() / 'captures' / part_id_clean
            cap_dir = base / ts.strftime('%Y-%m-%d') / ts.strftime('%H%M%S')
            cap_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            cap_dir = None
        # Remember capture directory for cycle time logging
        try:
            self._cycle_cap_dir = cap_dir
        except Exception:
            pass
        # Offer to open tuner first if no saved contour params yet
        try:
            st = state()
            if getattr(st, 'contour_params', None) is None:
                resp = QMessageBox.question(self, "Edge/Contour Tuner", "Open tuner to calibrate edge/contour before detection?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if resp == QMessageBox.Yes:
                    self.on_open_tuner()
        except Exception:
            pass
        # Require project
        try:
            if not self._ensure_models_loaded(required=("top", "front", "defect"), show_dialog=True):
                return
        except Exception:
            pass
        # Prefer top camera capture; fallback to currently loaded image
        img_path = None
        from services import camera_service
        try:
            if camera_service.is_connected("Top"):
                # Capture and write to captures folder for Detectron
                frame = cammgr.capture("Top")
                # update preview immediately
                pm = np_bgr_to_qpixmap(frame)
                self.preview_panel.set_original_np(pm)
                # Save raw capture into captures folder
                if cap_dir is not None:
                    try:
                        raw_path = str((cap_dir / 'step-01_top_raw.png'))
                        cv2.imwrite(raw_path, frame)
                        self.workflow_tab.append_log(f"[Capture] Saved raw: {raw_path}")
                        img_path = raw_path
                    except Exception:
                        pass
                # If we couldn't save to captures, write to a known temporary file and pass that
                if not img_path:
                    try:
                        tmp = os.path.join(os.getcwd(), 'top_capture.png')
                        cv2.imwrite(tmp, frame)
                        img_path = tmp
                    except Exception:
                        pass
            else:
                img_path = self._current_image_path
                # If using a file, copy it into captures for record
                if cap_dir is not None and img_path:
                    try:
                        import shutil, os
                        if os.path.exists(img_path):
                            dst = str((cap_dir / 'step-01_attachment_source.png'))
                            shutil.copyfile(img_path, dst)
                            self.workflow_tab.append_log(f"[Capture] Copied source: {dst}")
                            img_path = dst
                    except Exception:
                        pass
        except Exception as ex:
            self.workflow_tab.append_log(f"[Camera] Capture failed: {ex}")
            img_path = self._current_image_path
        if not img_path:
            QMessageBox.information(self, "Run Detection", "Please connect top camera or load an image first.")
            return
        self.workflow_tab.append_log("[Detectron] Running detection...")
        try:
            self.workflow_tab.append_log(f"[Detectron] Detecting file: {img_path}")
        except Exception:
            pass
        try:
            # keep last processed path for tuning/preview (even if detect raises later)
            try:
                self._last_capture_path = img_path
            except Exception:
                pass
            results = solvision_manager.detect(img_path)
            self.workflow_tab.append_log(f"[Detectron] Step 1 returned {len(results)} detection(s)")
            # Compute arrows + CCW numbering (counterclockwise) starting at bottom-right
            try:
                import cv2 as _cv2, math
                from services import contour_tools as _ct
                src_for_arrows = _cv2.imread(img_path)
                if src_for_arrows is not None:
                    try:
                        from services.config import state as _state
                        _p = getattr(_state(), "contour_params", None) or _ct.DEFAULT_PARAMS
                    except Exception:
                        _p = _ct.DEFAULT_PARAMS
                    arrows, contour = _ct.compute_arrows_for_detections(src_for_arrows, results, params=_p)
                    for det, arr in zip(results, arrows):
                        try:
                            if isinstance(arr, dict) and arr.get('anchor') and arr.get('vec'):
                                det['arrow'] = arr
                        except Exception:
                            pass
                    # Reference is exact image center (turntable center)
                    h, w = src_for_arrows.shape[:2]
                    cx0, cy0 = w/2.0, h/2.0
                    pts = []
                    for d in results:
                        b = d.get('bounds')
                        if not b:
                            continue
                        x,y,w,h = b
                        cx = x + w/2.0; cy = y + h/2.0
                        # Store both detection center and image center for clarity
                        d['det_center'] = (cx, cy)
                        d['center'] = (cx0, cy0)      # image center
                        d['center_ref'] = (cx0, cy0)
                        d['offset_top_px'] = cx - cx0
                        ang = math.atan2(cy0 - cy, cx - cx0)  # 0 at right, CCW positive
                        pts.append((cx, cy, ang, d))
                    if pts:
                        # Start with the detection closest to bottom-right (315 deg) and walk CCW
                        ref = -math.pi * 0.25  # -45 degrees
                        def _ang_diff(a, b=ref):
                            r = a - b
                            while r <= -math.pi: r += 2*math.pi
                            while r > math.pi: r -= 2*math.pi
                            return abs(r)
                        start = min(pts, key=lambda t: _ang_diff(t[2]))
                        ang0 = start[2]
                        def _rank(p):
                            r = p[2] - ang0
                            while r < 0: r += 2*math.pi
                            while r >= 2*math.pi: r -= 2*math.pi
                            return r
                        pts.sort(key=_rank)
                        for i, (_,_,_, d) in enumerate(pts, start=1):
                            d['index'] = i
            except Exception as ex:
                self.workflow_tab.append_log(f"[Detectron] Arrow computation skipped: {ex}")
            self.workflow_tab.populate_detection_results(results)
            # Compute phi for each detection from arrow vector relative to vertical (CW positive, CCW negative)
            try:
                import math
                for d in results:
                    try:
                        vec = d.get('arrow', {}).get('vec')
                        if vec is not None:
                            vx, vy = float(vec[0]), float(vec[1])
                            ang_from_vertical_to_v = math.atan2(vx, -vy)
                            d['phi'] = -ang_from_vertical_to_v
                            # Also compute rotated horizontal offset in top view after applying phi
                            try:
                                (cx, cy) = d.get('det_center') or (None, None)
                                (cx0, cy0) = d.get('center_ref') or (None, None)
                                if cx is not None and cx0 is not None:
                                    dx = float(cx - cx0); dy = float(cy - cy0)
                                    # Exact requested formula (CCW by phi):
                                    # x' = x0 + (x-x0)*cos(phi) - (y-y0)*sin(phi)
                                    # store x' - x0
                                    x_rot_off = dx * math.cos(d['phi']) - dy * math.sin(d['phi'])
                                    d['offset_top_rot_px'] = x_rot_off
                                    # Optional verbose log per detection
                                    try:
                                        idx_val = d.get('index')
                                        off0 = d.get('offset_top_px')
                                        phi_deg = math.degrees(d.get('phi') or 0.0)
                                        self.workflow_tab.append_log(
                                            f"[Step1] idx={idx_val} det_center=({cx:.1f},{cy:.1f}) img_center=({cx0:.1f},{cy0:.1f}) "
                                            f"phi={phi_deg:.2f}deg off_top_px={off0:.1f} off_top_rot_px={x_rot_off:.1f}"
                                        )
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                    except Exception:
                        pass
            except Exception:
                pass
            self.defect_ledger.populate_ledger(results)
            self.workflow_tab.append_log(f"[Detectron] {len(results)} detection(s)")
            try:
                # Arrows computed above; overlay arrows only
                self.preview_panel.set_draw_boxes(False)
                self.preview_panel.set_attachment_detections(results)
                # Save annotated view if capture dir is available
                if cap_dir is not None:
                    try:
                        out_path = str((cap_dir / 'step-01_top_annotated.png'))
                        if self.preview_panel.save_attachment_view(out_path):
                            self.workflow_tab.append_log(f"[Capture] Saved annotated: {out_path}")
                    except Exception:
                        pass
                # Also save cropped images for each detection and annotate with blue arrows
                if cap_dir is not None:
                    try:
                        from pathlib import Path as _Path
                        import cv2 as _cv2
                        import numpy as _np
                        import re as _re
                        from services import contour_tools as _ct
                        crops_dir = _Path(cap_dir) / 'step-01 cropped images'
                        crops_dir.mkdir(parents=True, exist_ok=True)
                        src = _cv2.imread(img_path)
                        if src is None:
                            self.workflow_tab.append_log("[Capture] Skipped crops: could not read source image")
                        else:
                            ih, iw = src.shape[:2]
                            # Use established global indices order
                            items = []
                            for det in results:
                                try:
                                    items.append((int(det.get('index', 0)), det))
                                except Exception:
                                    items.append((0, det))
                            items.sort(key=lambda t: t[0])

                            saved = 0
                            for _, det in items:
                                idx = int(det.get('index', 0)) or 0
                                b = det.get('bounds')
                                if not b:
                                    continue
                                x, y, w, h = b
                                # Validate and clamp to image bounds
                                try:
                                    x = int(max(0, x or 0)); y = int(max(0, y or 0))
                                    w = int(max(1, w or 0)); h = int(max(1, h or 0))
                                except Exception:
                                    continue
                                if x >= iw or y >= ih:
                                    continue
                                x2 = min(iw, x + w); y2 = min(ih, y + h)
                                cw, ch = x2 - x, y2 - y
                                if cw <= 0 or ch <= 0:
                                    continue
                                crop = src[y:y2, x:x2].copy()

                                # Use global arrow vector (anchor/vec) and map to crop coords
                                arr = det.get('arrow') if isinstance(det, dict) else None
                                if isinstance(arr, dict) and arr.get('anchor') and arr.get('vec'):
                                    anc = arr['anchor']; vec = arr['vec']
                                    ax = int(anc[0] - x); ay = int(anc[1] - y)
                                    ex = int(anc[0] + vec[0] - x); ey = int(anc[1] + vec[1] - y)
                                    _cv2.arrowedLine(crop, (ax, ay), (ex, ey), (255, 0, 0), 2, tipLength=0.30)
                                    # phi label on crop (yellow)
                                    phi = det.get('phi')
                                    if isinstance(phi, (int, float)):
                                        _cv2.putText(crop, f"{phi:.3f}", (ex + 6, ey), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                # Put index top-left
                                _cv2.putText(crop, f"{idx}", (5, 18), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                                # Build filename: det_###_{class}_{score}.png, ordered right->left
                                label = det.get('class')
                                score = det.get('score')
                                try:
                                    score_str = f"{float(score):.2f}" if score is not None else None
                                except Exception:
                                    score_str = None
                                base = f"det_{idx:03d}"
                                if label:
                                    safe = _re.sub(r"[^A-Za-z0-9._-]+", "_", str(label))
                                    base += f"_{safe}"
                                if score_str:
                                    base += f"_{score_str}"
                                out_file = crops_dir / f"{base}.png"
                                try:
                                    _cv2.imwrite(str(out_file), crop)
                                    saved += 1
                                except Exception:
                                    pass
                            self.workflow_tab.append_log(f"[Capture] Saved {saved} cropped image(s) with arrows to: {str(crops_dir)}")
                    except Exception as ex:
                        self.workflow_tab.append_log(f"[Capture] Crops skipped: {ex}")
                # Step 2: rotate per phi and capture front images
                if not results:
                    self.workflow_tab.append_log("[Step2] Skipping rotation: no detections returned.")
                elif cap_dir is None:
                    self.workflow_tab.append_log("[Step2] Skipping rotation: capture directory unavailable.")
                elif not linear_axis_service.is_calibrated():
                    self.workflow_tab.append_log("[Step2] Skipping rotation: linear axis not calibrated. Please connect and calibrate the linear axis first.")
                else:
                    try:
                        self.workflow_tab.append_log("[Step2] Starting rotation sequence...")
                        self._run_step2_sequence(results, cap_dir)
                    except Exception as ex:
                        self.workflow_tab.append_log(f"[Step2] Failed to start: {ex}")
            except Exception:
                pass
        except Exception as ex:
            QMessageBox.warning(self, "Detection", f"Detection failed.\n{ex}")
        finally:
            pass

    def on_run_step3_step4_existing(self):
        """Run only step 3/4 on a previously captured run (step-02 crops)."""
        from pathlib import Path
        import threading

        if not self._ensure_models_loaded(required=("front", "defect"), show_dialog=True):
            return

        base_dir = QFileDialog.getExistingDirectory(self, "Select Run Folder", "")
        if not base_dir:
            return
        step2_dir = Path(base_dir) / "step-02"
        crops_dir = step2_dir / "step_2_cropped"
        if not step2_dir.exists() or not crops_dir.exists():
            QMessageBox.information(
                self,
                "Run Step 3/4",
                "Selected folder does not contain step-02/step_2_cropped.\n"
                "Please select a run directory that has existing step-02 crops.",
            )
            return

        def run():
            try:
                self.tt_message.emit(f"[Step3/4] Using existing run folder: {base_dir}")
            except Exception:
                pass
            try:
                self._run_step3_front(step2_dir)
            except Exception as ex:
                try:
                    self.tt_message.emit(f"[Step3] Failed: {ex}")
                except Exception:
                    pass
            try:
                self._run_step4_defect(step2_dir)
            except Exception as ex:
                try:
                    self.tt_message.emit(f"[Step4] Failed: {ex}")
                except Exception:
                    pass
            try:
                self.tt_message.emit("[Step3/4] Completed processing existing run.")
            except Exception:
                pass

        threading.Thread(target=run, daemon=True).start()

    def on_open_tuner(self):
        try:
            from .edge_tuner import EdgeTunerDialog
        except Exception as ex:
            QMessageBox.warning(self, "Edge/Contour Tuner", f"Could not open tuner: {ex}")
            return
        dlg = EdgeTunerDialog(self)
        # Prefill with saved contour params if available
        try:
            st = state(); p = getattr(st, "contour_params", None)
            if isinstance(p, dict) and p:
                try:
                    if p.get("method"):
                        idx = dlg.combo_method.findText(str(p.get("method")))
                        if idx >= 0:
                            dlg.combo_method.setCurrentIndex(idx)
                    if "blur" in p: dlg.spin_blur.setValue(int(p.get("blur")))
                    if "morph" in p: dlg.spin_morph.setValue(int(p.get("morph")))
                    if "morph_iter" in p: dlg.spin_morph_iter.setValue(int(p.get("morph_iter")))
                    if "approx_eps" in p: dlg.spin_eps.setValue(float(p.get("approx_eps")))
                    if "thresh_offset" in p: dlg.spin_th_off.setValue(float(p.get("thresh_offset")))
                    if "smooth_iters" in p: dlg.spin_smooth.setValue(int(p.get("smooth_iters")))
                    if "arrow_len" in p: dlg.spin_arrow.setValue(float(p.get("arrow_len")))
                except Exception:
                    pass
        except Exception:
            pass
        # Prefer a live top-camera snapshot so tuning can be done before detection
        chosen_path = None
        try:
            from services import camera_service as _cam
            import cv2 as _cv2
            from pathlib import Path as _Path
            if _cam.is_connected("Top"):
                try:
                    frame = _cam.capture("Top")
                    # Show immediately in tuner (no need to save first)
                    try:
                        dlg.set_image_np(frame)
                    except Exception:
                        pass
                    # Also write a temp file for convenience/inspection
                    try:
                        base = app_root()
                        tmp_path = str(base / "tuner_top.png")
                        _cv2.imwrite(tmp_path, frame)
                        chosen_path = tmp_path
                    except Exception:
                        chosen_path = None
                except Exception:
                    chosen_path = None
        except Exception:
            chosen_path = None
        if not chosen_path:
            # Fallback to latest capture or user-selected image
            try:
                chosen_path = getattr(self, "_last_capture_path", None)
            except Exception:
                chosen_path = None
        if not chosen_path:
            try:
                chosen_path = getattr(self, "_current_image_path", None)
            except Exception:
                chosen_path = None
        if chosen_path:
            if not dlg.set_image_path(chosen_path):
                QMessageBox.information(self, "Edge/Contour Tuner", f"Could not load image:\n{chosen_path}\nUse 'Open Image…' to pick a file.")
        else:
            # No camera, no capture, no selected image
            self.workflow_tab.append_log("[Tuner] No top camera/capture/image; please choose a file.")
        # Optionally auto-run a preview so users see immediate feedback
        try:
            from PyQt5.QtCore import QTimer as _QTimer
            _QTimer.singleShot(0, dlg._preview_contour)
        except Exception:
            pass
        if dlg.exec_() == dlg.Accepted:
            # Apply arrows with current tuner params on the latest capture and refresh overlay
            try:
                # Persist current tuner params for future detections, even without a capture
                try:
                    st = state(); st.contour_params = dict(dlg.params()); save_state()
                except Exception:
                    pass
                img_path = getattr(self, "_last_capture_path", None)
                if not img_path:
                    return
                from services import contour_tools as _ct
                import cv2 as _cv2
                src = _cv2.imread(img_path)
                if src is None:
                    return
                # Re-run detect quickly; compute arrows + CCW indices using contour
                results = solvision_manager.detect(img_path)
                import math
                arrows, contour = _ct.compute_arrows_for_detections(src, results, params=dlg.params())
                # Reference is exact image center (turntable center)
                h, w = src.shape[:2]
                cx0, cy0 = w/2.0, h/2.0
                pts = []
                for d in results:
                    b = d.get('bounds')
                    if not b:
                        continue
                    x,y,w,h = b
                    cx = x + w/2.0; cy = y + h/2.0
                    d['det_center'] = (cx, cy)
                    d['center'] = (cx0, cy0)
                    d['offset_top_px'] = cx - cx0
                    ang = math.atan2(cy0 - cy, cx - cx0)
                    pts.append((cx, cy, ang, d))
                if pts:
                    # Start with the detection closest to bottom-right (315 deg) and walk CCW
                    ref = -math.pi * 0.25  # -45 degrees
                    def _ang_diff(a, b=ref):
                        r = a - b
                        while r <= -math.pi: r += 2*math.pi
                        while r > math.pi: r -= 2*math.pi
                        return abs(r)
                    start = min(pts, key=lambda t: _ang_diff(t[2]))
                    ang0 = start[2]
                    def _rank(p):
                        r = p[2] - ang0
                        while r < 0: r += 2*math.pi
                        while r >= 2*math.pi: r -= 2*math.pi
                        return r
                    pts.sort(key=_rank)
                    for i, (_,_,_, d) in enumerate(pts, start=1):
                        d['index'] = i
                for det, arr in zip(results, arrows):
                    try:
                        if isinstance(arr, dict) and arr.get('anchor') and arr.get('vec'):
                            det['arrow'] = arr
                    except Exception:
                        pass
                # Update preview overlay only
                self.preview_panel.set_draw_boxes(False)
                self.preview_panel.set_attachment_detections(results)
                self.workflow_tab.append_log("[Tuner] Applied contour params to overlay.")
            except Exception as ex:
                QMessageBox.information(self, "Tuner", f"Preview apply failed: {ex}")

    # Blob tuner and backend switching removed

    def on_overlay_toggled(self, checked: bool):
        self.preview_panel.set_overlay_enabled(checked)
        self.workflow_tab.append_log(f"Overlay toggled: {'ON' if checked else 'OFF'}")
        st = state(); st.overlay_enabled = bool(checked); save_state()

    def on_prev(self):
        # Placeholder: gallery navigation not implemented
        self.workflow_tab.append_log("Navigate to previous inspection (placeholder)")

    def on_next(self):
        # Placeholder: gallery navigation not implemented
        self.workflow_tab.append_log("Navigate to next inspection (placeholder)")

    # Turntable slots
    def on_turntable_refresh(self):
        try:
            ports = turntable_service.refresh_devices()
            self.workflow_tab.turntable_panel.set_ports(ports)
        except Exception as ex:
            self.workflow_tab.append_log(f"[Turntable] Refresh failed: {ex}")

    def on_turntable_connect(self, port: str):
        ok = turntable_service.connect(port)
        if ok:
            self.workflow_tab.turntable_panel.set_connected(True, port)
            self.workflow_tab.append_log(f"[Turntable] Connected to {port}.")
            st = state(); st.turntable_port = port; st.turntable_step = float(self.workflow_tab.turntable_panel.step.value()); save_state()
        else:
            self.workflow_tab.append_log(f"[Turntable] Connection failed for {port}.")

    def on_turntable_disconnect(self):
        turntable_service.disconnect()
        self.workflow_tab.turntable_panel.set_connected(False)
        self.workflow_tab.append_log("[Turntable] Disconnected.")

    def on_turntable_home(self):
        # Run homing in a background thread to avoid blocking UI
        import threading

        def run():
            res = turntable_service.home()
            self.tt_message.emit(res.message)
            status = res.message if res.success else f"Error: {res.message}"
            self.tt_status.emit(status)

        threading.Thread(target=run, daemon=True).start()

    # ---- Step 2 pipeline: rotate per phi and capture front images ----
    def _run_step2_sequence(self, detections, cap_dir):
        import threading, os, time
        from pathlib import Path
        import math
        from services import turntable_service
        from services import camera_service
        from .qt_image import np_bgr_to_qpixmap
        from PyQt5.QtCore import QTimer

        # Prepare folder
        step2_dir = Path(cap_dir) / 'step-02'
        try:
            step2_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        step3_dir = step2_dir.parent / 'step-03'
        step4_dir = step2_dir.parent / 'step-04'
        for _d in (step3_dir, step4_dir):
            try:
                _d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        # Order by index
        ordered = []
        try:
            ordered = sorted([d for d in detections if isinstance(d, dict)], key=lambda d: int(d.get('index', 0)))
        except Exception:
            ordered = list(detections)

        def worker():
            try:
                cycle_start = float(getattr(self, "_cycle_start_ts", None) or time.time())
            except Exception:
                cycle_start = time.time()
            # Check turntable
            if not turntable_service.is_connected():
                self.tt_message.emit("[Step2] Turntable not connected; skipping rotation sequence.")
                return
            # Require models to be explicitly loaded by the user
            front_model = solvision_manager.current_project_path_for('front')
            defect_model = solvision_manager.current_project_path_for('defect')
            try:
                from services.config import state as _state
                st_def = _state()
                self._defect_thr_cached = getattr(st_def, "defect_score_threshold", None)
            except Exception:
                self._defect_thr_cached = None
            if not front_model:
                self.tt_message.emit("[Step2] Front model not loaded; skipping rotation/front alignment.")
                return

            # Background executor to overlap step-03/04 with motions (single worker to keep Detectron safe)
            exec_bg = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            bg_futures = []

            def _submit_step4(bbox_path, idx):
                if not bbox_path or not defect_model:
                    return
                try:
                    f = exec_bg.submit(
                        self._process_step4_single,
                        bbox_path,
                        idx,
                        step4_dir,
                        defect_model,
                        self._defect_thr_cached,
                    )
                    bg_futures.append(f)
                except Exception as ex:
                    try:
                        self.tt_message.emit(f"[Step4] idx {idx}: submit failed: {ex}")
                    except Exception:
                        pass

            def _submit_step3(crop_path, idx):
                if not crop_path or not front_model:
                    return
                try:
                    f = exec_bg.submit(self._process_step3_single, crop_path, idx, step3_dir, front_model)
                    bg_futures.append(f)
                except Exception as ex:
                    try:
                        self.tt_message.emit(f"[Step3] idx {idx}: submit failed: {ex}")
                    except Exception:
                        pass
                    return

                def _on_done(fut, _idx=idx):
                    try:
                        bbox_path = None
                        try:
                            bbox_path = fut.result()
                        except Exception as inner_ex:
                            try:
                                self.tt_message.emit(f"[Step3] idx {_idx}: failed: {inner_ex}")
                            except Exception:
                                pass
                            return
                        _submit_step4(bbox_path, _idx)
                    except Exception:
                        pass

                try:
                    f.add_done_callback(_on_done)
                except Exception:
                    pass
            # Snapshot helper (post to UI thread)
            def _show_front(frame):
                try:
                    QTimer.singleShot(0, lambda f=frame: self.preview_panel.set_front_np(np_bgr_to_qpixmap(f)))
                except Exception:
                    pass
            def _show_top(frame):
                try:
                    QTimer.singleShot(0, lambda f=frame: self.preview_panel.set_original_np(np_bgr_to_qpixmap(f)))
                except Exception:
                    pass

            # Move sequentially by index, but turntable and axis move simultaneously per index.
            last_phi = 0.0
            for d in ordered:
                phi = d.get('phi')
                idx = int(d.get('index', 0) or 0)
                if not isinstance(phi, (int, float)):
                    self.tt_message.emit(f"[Step2] Skipping index {idx}: missing phi.")
                    continue

                # Compute turntable delta (deg) and desired axis target (mm) for this detection.
                move_deg = math.degrees(phi - last_phi)
                last_phi = phi

                target_mm = None
                axis_reason = None
                try:
                    if linear_axis_service.is_calibrated():
                        from services.config import state as _state
                        try:
                            off_top = float(d.get('offset_top_rot_px', d.get('offset_top_px', 0.0)) or 0.0)
                        except Exception:
                            off_top = 0.0
                        st_axis = _state()
                        top_fov_val = getattr(st_axis, "front_fov_top_px", None) or DEFAULT_FRONT_FOV_TOP_PX
                        try:
                            top_fov_val = float(top_fov_val)
                        except (TypeError, ValueError):
                            top_fov_val = 0.0
                        if abs(top_fov_val) > 1e-3:
                            PIXELS_PER_MM = 72.3035714
                            FRONT_WIDTH_PX = 2592.0
                            delta_px = (off_top / top_fov_val) * FRONT_WIDTH_PX
                            delta_mm = delta_px / PIXELS_PER_MM
                            tgt = 50.0 + delta_mm
                            target_mm = max(0.0, min(100.0, tgt))
                        else:
                            axis_reason = "invalid front FOV"
                    else:
                        axis_reason = "linear axis not calibrated"
                except Exception as ex:
                    axis_reason = f"axis alignment failed: {ex}"

                # Helpers for robust moves
                def _axis_move_safe(target: float) -> dict:
                    res = {"msg": None, "err": None}
                    try:
                        curr = linear_axis_service.current_position_mm()
                    except Exception:
                        curr = None
                    if curr is not None and abs(curr - target) < 0.05:
                        res["msg"] = "[INFO] Already at requested position."
                        return res
                    try:
                        move_res = linear_axis_service.goto_mm(target)
                        if move_res.success:
                            res["msg"] = move_res.message
                        else:
                            # Retry once if timed out
                            if "timed out" in move_res.message.lower():
                                retry = linear_axis_service.goto_mm(target)
                                if retry.success:
                                    res["msg"] = retry.message + " (retried)"
                                else:
                                    res["err"] = retry.message
                            else:
                                res["err"] = move_res.message
                    except Exception as ex:
                        res["err"] = str(ex)
                    return res

                def _tt_move_safe(delta_deg: float) -> dict:
                    res = {"msg": None, "err": None}
                    if abs(delta_deg) < 1e-3:
                        res["msg"] = "[Turntable] Homing complete (already at zero)."
                        return res
                    try:
                        msg = turntable_service.move_relative(delta_deg)
                        res["msg"] = msg
                    except Exception as ex:
                        # Retry once
                        try:
                            msg = turntable_service.move_relative(delta_deg)
                            res["msg"] = msg + " (retried)"
                        except Exception as ex2:
                            res["err"] = str(ex2)
                    return res

                # Fire moves concurrently
                import threading
                tt_res = {"msg": None, "err": None}
                ax_res = {"msg": None, "err": axis_reason}

                def _move_tt():
                    r = _tt_move_safe(move_deg)
                    tt_res.update(r)

                def _move_axis():
                    if target_mm is None:
                        return
                    r = _axis_move_safe(target_mm)
                    ax_res.update(r)

                threads = []
                t1 = threading.Thread(target=_move_tt, daemon=True); threads.append(t1); t1.start()
                if target_mm is not None:
                    t2 = threading.Thread(target=_move_axis, daemon=True); threads.append(t2); t2.start()

                for t in threads:
                    t.join()

                # Log results
                if tt_res["err"]:
                    self.tt_message.emit(f"[Step2] Rotate idx {idx} failed: {tt_res['err']}")
                    continue
                if tt_res["msg"]:
                    self.tt_message.emit(f"[Step2] Rotate idx {idx}: {tt_res['msg']}")

                if target_mm is None:
                    if axis_reason:
                        self.tt_message.emit(f"[Step2] Axis alignment skipped: {axis_reason}")
                else:
                    if ax_res["err"]:
                        self.tt_message.emit(f"[Step2] Axis alignment failed: {ax_res['err']}")
                    elif ax_res["msg"]:
                        self.tt_message.emit(ax_res["msg"])

                # Capture from cameras if available and update previews
                try:
                    top_snapshot = None
                    # Top camera preview update (keep latest frame for debugging)
                    if cammgr.is_connected("Top"):
                        try:
                            top_frame = cammgr.capture("Top")
                            top_snapshot = self._ensure_bgr8(top_frame)
                            _show_top(top_snapshot)
                        except Exception as ex:
                            top_snapshot = None
                            self.tt_message.emit(f"[Step2] Top snapshot failed: {ex}")
                    # Front camera preview, detect, correct, and crop
                    if cammgr.is_connected("Front"):
                        import cv2 as _cv2
                        import numpy as _np
                        from services.config import state as _state

                        def _capture_front():
                            frame = cammgr.capture("Front")
                            return self._ensure_bgr8(frame)

                        def _center_crop(img, crop_size):
                            Hc, Wc = img.shape[:2]
                            half = crop_size // 2
                            cx = Wc // 2
                            cy = Hc // 2
                            x0 = max(0, cx - half); x1 = min(Wc, cx + half)
                            y0 = max(0, cy - half); y1 = min(Hc, cy + half)
                            crop = img[y0:y1, x0:x1].copy()
                            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                                crop = _cv2.resize(crop, (crop_size, crop_size))
                            return crop

                        # first capture at current alignment
                        overlay = _capture_front()
                        initial_raw_path = None
                        try:
                            initial_raw_path = str(step2_dir / f"step-02_front_initial_{idx:03d}.png")
                            _cv2.imwrite(initial_raw_path, overlay)
                            self.tt_message.emit(f"[Step2] Saved initial front snapshot: {initial_raw_path}")
                        except Exception:
                            initial_raw_path = None

                        try:
                            st2 = _state(); crop_size = int(getattr(st2, 'step2_crop_size', None) or 1600)
                        except Exception:
                            crop_size = 1600
                        crop = _center_crop(overlay, crop_size)
                        initial_crop_path = str(step2_dir / f"step-02_front_crop_initial_{idx:03d}.png")
                        try:
                            _cv2.imwrite(initial_crop_path, crop)
                        except Exception:
                            pass

                        # Run front detection on the initial crop
                        dets = []
                        try:
                            dets = solvision_manager.detect_for('front', initial_crop_path)
                        except Exception as ex:
                            self.tt_message.emit(f"[Step2] Front detect failed: {ex}")
                            dets = []

                        if not dets:
                            self.tt_message.emit(f"[Step2] No detection in crop idx {idx}; discarding filling.")
                            continue

                        # Pick detection closest to crop center
                        cx_crop = crop.shape[1] / 2.0
                        cy_crop = crop.shape[0] / 2.0
                        def _center_of(det):
                            b = det.get("bounds")
                            if not b:
                                return (None, None)
                            x1, y1, w, h = b
                            return (x1 + w / 2.0, y1 + h / 2.0)
                        det = min(dets, key=lambda dd: float('inf') if _center_of(dd)[0] is None else abs(_center_of(dd)[0] - cx_crop) + abs(_center_of(dd)[1] - cy_crop))
                        dcx, dcy = _center_of(det)
                        if dcx is None:
                            self.tt_message.emit(f"[Step2] Detection missing center; discarding idx {idx}.")
                            continue
                        dx_px = dcx - cx_crop  # + => bbox to the right of center
                        # Convert pixel offset to mm using front camera scale
                        PIXELS_PER_MM = 66.3035714
                        dx_mm = dx_px / PIXELS_PER_MM
                        try:
                            curr_pos = linear_axis_service.current_position_mm()
                            if curr_pos is None:
                                curr_pos = 50.0
                        except Exception:
                            curr_pos = 50.0
                        # Flip sign: bbox right of center -> move axis left (negative delta)
                        new_target = max(0.0, min(100.0, curr_pos - dx_mm))
                        if abs(dx_mm) > 0.05:
                            try:
                                corr_res = _axis_move_safe(new_target)
                                if corr_res["err"]:
                                    self.tt_message.emit(f"[Step2] Correction move failed: {corr_res['err']}")
                                elif corr_res["msg"]:
                                    self.tt_message.emit(f"{corr_res['msg']} (correction dx={dx_px:.2f}px -> {dx_mm:.2f}mm, new={new_target:.2f}mm)")
                            except Exception as ex:
                                self.tt_message.emit(f"[Step2] Correction move failed: {ex}")
                        else:
                            self.tt_message.emit(f"[Step2] Alignment within tolerance (dx={dx_px:.2f}px); no correction move.")

                        # Capture corrected frame
                        overlay = _capture_front()
                        corrected_raw_path = None
                        try:
                            corrected_raw_path = str(step2_dir / f"step-02_front_corrected_{idx:03d}.png")
                            _cv2.imwrite(corrected_raw_path, overlay)
                            self.tt_message.emit(f"[Step2] Saved corrected front snapshot: {corrected_raw_path}")
                        except Exception:
                            corrected_raw_path = None

                        H, W = overlay.shape[:2]
                        x_mark = W // 2
                        midy = H // 2
                        try:
                            _cv2.circle(overlay, (x_mark, midy), 8, (255, 0, 0), -1)
                            _cv2.circle(overlay, (x_mark, midy), 8, (255, 255, 255), 2)
                        except Exception:
                            pass

                        # Update front preview
                        try:
                            pm_front = np_bgr_to_qpixmap(overlay)
                            _show_front(pm_front)
                            try:
                                from PyQt5.QtCore import QTimer as _QTimer
                                _QTimer.singleShot(0, lambda x=x_mark: self.preview_panel.set_front_markers([x]))
                            except Exception:
                                pass
                            QTimer.singleShot(0, lambda det=[]: self.preview_panel.set_front_detections(det))
                        except Exception:
                            pass

                        # Save annotated and crop corrected center for downstream step 3
                        try:
                            fn_front = str(step2_dir / f"step-02_front_{idx:03d}.png")
                            if _cv2.imwrite(fn_front, overlay):
                                self.tt_message.emit(f"[Step2] Saved front snapshot (annotated): {fn_front}")
                            else:
                                self.tt_message.emit(f"[Step2] Failed to save front snapshot: {fn_front}")
                        except Exception as ex:
                            self.tt_message.emit(f"[Step2] Save failed: {ex}")

                        try:
                            crops_dir = step2_dir / 'step_2_cropped'
                            crops_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                st2 = _state(); crop_size = int(getattr(st2, 'step2_crop_size', None) or 1600)
                            except Exception:
                                crop_size = 1600
                            raw_img = None
                            try:
                                if corrected_raw_path:
                                    raw_img = _cv2.imread(corrected_raw_path)
                            except Exception:
                                raw_img = None
                            if raw_img is None:
                                raw_img = overlay.copy()
                            crop_final = _center_crop(raw_img, crop_size)
                            out_path = str(crops_dir / f"step-02_front_crop_{idx:03d}.png")
                            _cv2.imwrite(out_path, crop_final)
                            self.tt_message.emit(f"[Step2] Saved corrected crop: {out_path}")
                            try:
                                _submit_step3(out_path, idx)
                            except Exception:
                                pass
                        except Exception as ex:
                            self.tt_message.emit(f"[Step2] Crop failed: {ex}")

                        # Clear preview markers so the next filling starts clean
                        try:
                            from PyQt5.QtCore import QTimer as _QTimer
                            _QTimer.singleShot(0, lambda: self.preview_panel.set_front_markers([]))
                        except Exception:
                            pass

                        # Save latest top snapshot alongside the front capture if available
                        if top_snapshot is not None:
                            try:
                                fn_top = str(step2_dir / f"step-02_top_{idx:03d}.png")
                                if _cv2.imwrite(fn_top, top_snapshot):
                                    self.tt_message.emit(f"[Step2] Saved top snapshot: {fn_top}")
                                else:
                                    self.tt_message.emit(f"[Step2] Failed to save top snapshot: {fn_top}")
                            except Exception as ex:
                                self.tt_message.emit(f"[Step2] Top save failed: {ex}")

                        _show_front(overlay)
                    else:
                        self.tt_message.emit("[Step2] Front camera not connected; snapshot skipped.")
                except Exception as ex:
                    self.tt_message.emit(f"[Step2] Snapshot failed: {ex}")

            # Wait for any pipelined Step3/4 tasks; fall back to sequential if none were scheduled
            try:
                if bg_futures:
                    try:
                        while True:
                            snapshot = list(bg_futures)
                            pending = [f for f in snapshot if not f.done()]
                            if not pending:
                                break
                            concurrent.futures.wait(pending, return_when=concurrent.futures.ALL_COMPLETED)
                    except Exception:
                        pass
                    for fut in list(bg_futures):
                        try:
                            fut.result()
                        except Exception as ex:
                            try:
                                self.tt_message.emit(f"[Step2] Background task failed: {ex}")
                            except Exception:
                                pass
                else:
                    # Backward-compatible sequential processing
                    try:
                        self._run_step3_front(step2_dir)
                    except Exception as ex:
                        self.tt_message.emit(f"[Step3] Failed: {ex}")
                    try:
                        self._run_step4_defect(step2_dir)
                    except Exception as ex:
                        self.tt_message.emit(f"[Step4] Failed: {ex}")
            finally:
                try:
                    exec_bg.shutdown(wait=True)
                except Exception:
                    pass
            # Fallback: ensure every bbox in step-03 has a step-04 result
            try:
                bbox_files = sorted(step3_dir.glob('step-03_front_bbox_*.png'))
                for p in bbox_files:
                    try:
                        import re as _re
                        m = _re.search(r"_(\d+)\.png$", p.name)
                        idx_fallback = int(m.group(1)) if m else 0
                    except Exception:
                        idx_fallback = 0
                    expected = step4_dir / f"step-04_defect_{idx_fallback:03d}.png"
                    if expected.exists():
                        continue
                    try:
                        self.tt_message.emit(f"[Step4] Fallback running idx {idx_fallback} from {p.name}")
                    except Exception:
                        pass
                    try:
                        self._process_step4_single(str(p), idx_fallback, step4_dir, defect_model, self._defect_thr_cached)
                    except Exception as ex:
                        try:
                            self.tt_message.emit(f"[Step4] Fallback idx {idx_fallback} failed: {ex}")
                        except Exception:
                            pass
            except Exception:
                pass
            # Final sweep: rerun step-04 sequentially over all bboxes to guarantee outputs
            try:
                self._run_step4_defect(step2_dir)
            except Exception as ex:
                try:
                    self.tt_message.emit(f"[Step4] Final sweep failed: {ex}")
                except Exception:
                    pass
            # Home the turntable at the end
            try:
                res = turntable_service.home()
                self.tt_message.emit(res.message)
                status = res.message if res.success else f"Error: {res.message}"
                self.tt_status.emit(status)
            except Exception:
                self.tt_message.emit("[Step2] Home failed.")
            # Home linear axis using configured home position
            try:
                if linear_axis_service.is_connected() and linear_axis_service.is_calibrated():
                    try:
                        from services.config import state as _state
                        hm = getattr(_state(), "linear_axis_home_mm", None)
                        home_mm = float(hm) if hm is not None else 50.0
                    except Exception:
                        home_mm = 50.0
                    res_ax_home = linear_axis_service.home(home_mm=home_mm)
                    self.tt_message.emit(res_ax_home.message)
                else:
                    self.tt_message.emit("[Step2] Axis home skipped (not connected/calibrated).")
            except Exception as ex:
                self.tt_message.emit(f"[Step2] Axis home failed: {ex}")
            # Record cycle time from button press to post-home
            try:
                elapsed = time.time() - cycle_start
                ct_path = Path(cap_dir) / "cycle_time.txt"
                ct_path.parent.mkdir(parents=True, exist_ok=True)
                with ct_path.open("a", encoding="ascii") as f:
                    f.write(f"{elapsed:.2f}\n")
                self.tt_message.emit(f"[Step2] Cycle time recorded: {elapsed:.2f} s -> {ct_path}")
            except Exception as ex:
                self.tt_message.emit(f"[Step2] Cycle time record failed: {ex}")

        threading.Thread(target=worker, daemon=True).start()

    # ---- Helpers ----
    def _ensure_bgr8(self, img):
        import numpy as _np
        import cv2 as _cv2

        if img is None:
            return _np.zeros((10, 10, 3), dtype=_np.uint8)

        arr = img
        if arr.ndim == 2:
            return _cv2.cvtColor(self._to_uint8(arr), _cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] == 1:
            single = self._to_uint8(arr[:, :, 0])
            return _cv2.cvtColor(single, _cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3 and arr.shape[2] >= 3:
            if arr.dtype == _np.uint8:
                return arr.copy()
            channels = [self._to_uint8(arr[:, :, c]) for c in range(3)]
            return _cv2.merge(channels)
        return arr

    def _to_uint8(self, channel):
        import numpy as _np
        arr = channel.astype(_np.float32, copy=False)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-6:
            return _np.zeros_like(channel, dtype=_np.uint8)
        norm = (arr - mn) / (mx - mn)
        return (norm * 255.0).clip(0, 255).astype(_np.uint8)

    def on_turntable_port_selected(self, port: str):
        st = state(); st.turntable_port = port; st.turntable_step = float(self.workflow_tab.turntable_panel.step.value()); save_state()

    def on_turntable_step_changed(self, v: float):
        st = state(); st.turntable_step = float(v); save_state()

    def on_defect_threshold_changed(self, v: float):
        try:
            st = state()
            st.defect_score_threshold = float(v)
            save_state()
            self.tt_message.emit(f"[Step4] Defect threshold updated to {float(v):.3f}")
        except Exception:
            pass

    def _ensure_models_loaded(self, required=("top",), show_dialog=False) -> bool:
        missing = []
        try:
            for name in required:
                if solvision_manager.current_project_path_for(name) is None:
                    missing.append(name)
        except Exception:
            missing = list(required)
        if missing:
            msg = f"Please load model(s): {', '.join(missing)} before running."
            try:
                if show_dialog:
                    QMessageBox.information(self, "Models Required", msg)
                else:
                    self.workflow_tab.append_log(msg)
            except Exception:
                pass
            return False
        return True

    def _on_tt_raw_message(self, msg: str):
        # Called from service thread; relay to UI thread via signal
        if msg:
            self.tt_message.emit(msg)

    def _handle_turntable_message(self, msg: str):
        self.workflow_tab.append_log(f"[Turntable] {msg}")

    def _handle_turntable_status(self, status: str):
        self.workflow_tab.turntable_panel.set_status(status)

    def on_turntable_rotate(self, angle: float):
        import threading

        def run():
            try:
                msg = turntable_service.move_relative(angle)
                self.tt_message.emit(msg)
                self.tt_status.emit(msg)
            except Exception as ex:
                self.tt_message.emit(f"Rotation failed: {ex}")
                self.tt_status.emit(str(ex))

        threading.Thread(target=run, daemon=True).start()

    # Linear axis handlers
    def _on_axis_raw_message(self, msg: str):
        if msg:
            try:
                from PyQt5.QtCore import QTimer as _QTimer
                # Queue log to UI thread to avoid cross-thread QTextEdit access
                _QTimer.singleShot(0, lambda m=msg: self.workflow_tab.append_log(f"[Axis] {m}"))
            except Exception:
                pass
            try:
                from PyQt5.QtCore import QTimer as _QTimer
                import re
                banner = msg.lower()
                # Mark controller ready once any banner/command text shows up
                if not self.workflow_tab.linear_axis_panel.is_ready():
                    if ("fuyu linear axis controller" in banner) or ("commands:" in banner) or ("stroke assumed" in banner):
                        self._axis_ui.set_ready.emit(True)
                # Detect calibration start/complete from device prints
                if "[cal]" in banner:
                    if "starting calibration" in banner:
                        self._axis_ui.set_calibrating.emit(True)
                    if "calibration complete" in banner or "right limit reached" in banner:
                        # Treat device message as authoritative end of calibration
                        self._axis_ui.set_calibrating.emit(False)
                        self._axis_ui.set_calibrated.emit(True, None)
                        self._axis_ui.set_ready.emit(True)
                # Detect calibration info output (R command)
                if "calibrated:" in banner:
                    pos_match = re.search(r"currentpos[:\s]+([-+]?\d+(?:\.\d+)?)", banner)
                    pos_val = None
                    if pos_match:
                        try:
                            pos_val = float(pos_match.group(1))
                        except Exception:
                            pos_val = None
                    if "yes" in banner:
                        def _apply_info(p=pos_val):
                            # Treat R output as authoritative end-of-calibration signal
                            self._axis_ui.set_calibrating.emit(False)
                            self._axis_ui.set_calibrated.emit(True, p)
                            self._axis_ui.set_ready.emit(True)

                        _apply_info()
                # Update position if reported
                if "[move] current position" in banner or "current position:" in banner or "currentpos" in banner:
                    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*mm", banner)
                    if m:
                        try:
                            pos = float(m.group(1))
                            self._axis_ui.set_position.emit(pos)
                        except Exception:
                            pass
            except Exception:
                pass

    def on_axis_refresh(self):
        try:
            ports = linear_axis_service.refresh_devices()
            self.workflow_tab.linear_axis_panel.set_ports(ports)
            st = state()
            if st.linear_axis_port:
                combo = self.workflow_tab.linear_axis_panel.port_combo
                idx = combo.findText(st.linear_axis_port)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        except Exception as ex:
            self.workflow_tab.append_log(f"[Axis] Refresh failed: {ex}")

    def on_axis_connect(self, port: str):
        from PyQt5.QtCore import QTimer as _QTimer
        if linear_axis_service.connect(port):
            _QTimer.singleShot(0, lambda: self.workflow_tab.linear_axis_panel.set_connected(True, port))
            _QTimer.singleShot(0, lambda: self.workflow_tab.linear_axis_panel.set_ready(False))
            _QTimer.singleShot(0, lambda: self.workflow_tab.append_log(f"[Axis] Connected to {port}."))
            st = state(); st.linear_axis_port = port; save_state()
            # Attach listener once connected
            try:
                linear_axis_service.add_listener(self._on_axis_raw_message)
            except Exception:
                pass
            # Fallback: if controller banner not seen soon, allow calibration anyway
            _QTimer.singleShot(1000, lambda: self.workflow_tab.linear_axis_panel.set_ready(True) if not self.workflow_tab.linear_axis_panel.is_ready() else None)
            # Apply persisted home value on connect
            try:
                home_mm = float(state().linear_axis_home_mm) if state().linear_axis_home_mm is not None else 50.0
                self.workflow_tab.linear_axis_panel.set_home_mm(home_mm)
            except Exception:
                pass
        else:
            _QTimer.singleShot(0, lambda: self.workflow_tab.append_log(f"[Axis] Connection failed for {port}."))

    def on_axis_disconnect(self):
        from PyQt5.QtCore import QTimer as _QTimer
        linear_axis_service.disconnect()
        _QTimer.singleShot(0, lambda: self.workflow_tab.linear_axis_panel.set_connected(False))
        _QTimer.singleShot(0, lambda: self.workflow_tab.append_log("[Axis] Disconnected."))

    def on_axis_calibrate(self):
        import threading

        # Guard against concurrent calibrations
        try:
            if self.workflow_tab.linear_axis_panel.is_calibrating():
                self.workflow_tab.append_log("[Axis] Calibration already in progress.")
                return
        except Exception:
            pass

        def run():
            self._axis_ui.set_calibrating.emit(True)
            try:
                try:
                    home_mm = float(self.workflow_tab.linear_axis_panel.home_mm())
                except Exception:
                    home_mm = 50.0
                res = linear_axis_service.calibrate_and_home(home_mm=home_mm)
                self.workflow_tab.append_log(res.message)
                pos = linear_axis_service.current_position_mm()
                if res.success:
                    # Optimistically unlock UI immediately after calibration/home succeeds
                    self._axis_ui.set_calibrated.emit(True, pos)
                    self._axis_ui.set_ready.emit(True)
                    try:
                        self.workflow_tab.append_log("[Axis] Reading calibration info (R)...")
                        info_res = linear_axis_service.read_calibration_info()
                    except Exception as info_ex:
                        info_res = None
                        self.workflow_tab.append_log(f"[Axis] Read calibration info failed: {info_ex}")
                    if info_res is not None:
                        self.workflow_tab.append_log(info_res.message)
                        pos_for_ui = info_res.position_mm if info_res.position_mm is not None else pos
                        if info_res.success:
                            self._axis_ui.set_calibrated.emit(True, pos_for_ui)
                            self._axis_ui.set_ready.emit(True)
                            try:
                                st = state(); st.linear_axis_last_mm = pos_for_ui; save_state()
                            except Exception:
                                pass
                        else:
                            self._axis_ui.set_calibrated.emit(True, pos_for_ui)
                            self._axis_ui.set_ready.emit(True)
                    else:
                        self._axis_ui.set_calibrated.emit(True, pos)
                        self._axis_ui.set_ready.emit(True)
                else:
                    self._axis_ui.set_calibrated.emit(False, pos)
            except Exception as ex:
                self.workflow_tab.append_log(f"[Axis] Calibration failed: {ex}")
            finally:
                self._axis_ui.set_calibrating.emit(False)

        threading.Thread(target=run, daemon=True).start()

    def on_axis_home(self, home_mm: float):
        import threading

        def run():
            try:
                res = linear_axis_service.home(home_mm=home_mm)
                self.workflow_tab.append_log(res.message)
                if res.success:
                    pos = linear_axis_service.current_position_mm()
                    self._axis_ui.set_calibrated.emit(True, pos)
                    try:
                        st = state(); st.linear_axis_last_mm = pos; save_state()
                    except Exception:
                        pass
            except Exception as ex:
                self.workflow_tab.append_log(f"[Axis] Home failed: {ex}")

        threading.Thread(target=run, daemon=True).start()

    def on_axis_goto(self, target_mm: float):
        import threading

        def run():
            try:
                res = linear_axis_service.goto_mm(target_mm)
                self.workflow_tab.append_log(res.message)
                if res.success:
                    pos = linear_axis_service.current_position_mm()
                    self._axis_ui.set_calibrated.emit(True, pos)
                    try:
                        st = state(); st.linear_axis_last_mm = pos; save_state()
                    except Exception:
                        pass
            except Exception as ex:
                self.workflow_tab.append_log(f"[Axis] Move failed: {ex}")

        threading.Thread(target=run, daemon=True).start()

    def on_axis_home_set(self, home_mm: float):
        try:
            st = state(); st.linear_axis_home_mm = float(home_mm); save_state()
            self.workflow_tab.append_log(f"[Axis] Home position set to {home_mm:.1f} mm.")
        except Exception:
            pass

    def _process_step3_single(self, crop_path, idx, step3_dir, front_path):
        import cv2 as _cv2
        from services import solvision_manager

        try:
            if not front_path:
                self.tt_message.emit("[Step3] No front_attachment model loaded; skipping.")
                return None
            if not os.path.isfile(crop_path):
                self.tt_message.emit(f"[Step3] idx {idx}: crop not found: {crop_path}")
                return None
        except Exception:
            return None

        try:
            img = _cv2.imread(str(crop_path))
            if img is None:
                self.tt_message.emit(f"[Step3] idx {idx}: failed to read {crop_path}")
                return None
            H, W = img.shape[:2]
            dets = []
            try:
                dets = solvision_manager.detect_for('front', str(crop_path))
            except Exception as ex:
                self.tt_message.emit(f"[Step3] idx {idx}: detect failed: {ex}")
                dets = []

            if not dets:
                ann = img.copy()
                _cv2.putText(ann, 'No detection', (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                out_ann = str(step3_dir / f"step-03_front_{idx:03d}.png")
                _cv2.imwrite(out_ann, ann)
                self.tt_message.emit(f"[Step3] idx {idx}: no detection; saved {out_ann}")
                return None

            cx0, cy0 = W * 0.5, H * 0.5

            def _metric(d):
                try:
                    b = d.get('bounds') or d.get('rect') or None
                    if not b or len(b) < 4:
                        return (float('inf'), -0.0)
                    x, y, w, h = b
                    x = float(x); y = float(y); w = float(w); h = float(h)
                    cx = x + w * 0.5; cy = y + h * 0.5
                    dist2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
                    sc = float(d.get('score') or 0.0)
                    return (dist2, -sc)
                except Exception:
                    return (float('inf'), -0.0)

            best = min(dets, key=_metric)
            bx, by, bw, bh = best.get('bounds') or (0, 0, 0, 0)
            try:
                bx = int(round(float(bx))); by = int(round(float(by)))
                bw = int(round(float(bw))); bh = int(round(float(bh)))
            except Exception:
                bx, by, bw, bh = 0, 0, 0, 0
            bx = max(0, min(W - 1, bx)); by = max(0, min(H - 1, by))
            bw = max(0, min(W - bx, bw)); bh = max(0, min(H - by, bh))

            ann = img.copy()
            def _safe_label_pos(x, y, w, h, text):
                (tw, th), _ = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                lx = max(0, min(W - tw - 1, x + 4))
                if y - th - 6 >= 0:
                    ly = y - 6
                else:
                    ly = min(H - 6, y + h + th)
                ly = max(th, min(H - 1, ly))
                return lx, ly
            def _hex_to_bgr(hex_str):
                try:
                    hs = hex_str.lstrip("#")
                    if len(hs) == 6:
                        r = int(hs[0:2], 16); g = int(hs[2:4], 16); b = int(hs[4:6], 16)
                        return (b, g, r)
                except Exception:
                    pass
                return (0, 255, 0)

            color = _hex_to_bgr(best.get("color"))
            _cv2.rectangle(ann, (bx, by), (bx + bw, by + bh), color, 2)
            label = str(best.get('class') or '')
            try:
                sc = best.get('score')
                if sc is not None:
                    label = f"{label} {float(sc):.2f}" if label else f"{float(sc):.2f}"
            except Exception:
                pass
            if label:
                lx, ly = _safe_label_pos(bx, by, bw, bh, label)
                _cv2.putText(ann, label, (lx, ly), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out_ann = str(step3_dir / f"step-03_front_{idx:03d}.png")
            _cv2.imwrite(out_ann, ann)

            pad = 50
            x0 = max(0, bx - pad)
            y0 = max(0, by - pad)
            x1 = min(W, bx + bw + pad)
            y1 = min(H, by + bh + pad)
            if x1 <= x0 or y1 <= y0:
                crop = img.copy()
            else:
                crop = img[y0:y1, x0:x1].copy()
            out_crop = str(step3_dir / f"step-03_front_bbox_{idx:03d}.png")
            _cv2.imwrite(out_crop, crop)
            self.tt_message.emit(f"[Step3] idx {idx}: saved {out_ann} and bbox {out_crop}")
            return out_crop
        except Exception as ex:
            try:
                self.tt_message.emit(f"[Step3] idx {idx}: failed: {ex}")
            except Exception:
                pass
            return None

    def _process_step4_single(self, bbox_path, idx, step4_dir, defect_path, override_thr=None):
        import cv2 as _cv2
        from services import solvision_manager

        try:
            if not defect_path:
                self.tt_message.emit("[Step4] No defect model loaded; skipping.")
                return
            if not os.path.isfile(bbox_path):
                self.tt_message.emit(f"[Step4] idx {idx}: bbox not found: {bbox_path}")
                return
        except Exception:
            return

        try:
            img = _cv2.imread(str(bbox_path))
            if img is None:
                self.tt_message.emit(f"[Step4] idx {idx}: failed to read {bbox_path}")
                return
            # Precompute palette once
            palette_bgr = []
            try:
                cols = solvision_manager.class_colors_for('defect')
                if cols:
                    for hs in cols:
                        try:
                            hs = str(hs).lstrip("#").strip()
                            if len(hs) == 6:
                                r = int(hs[0:2], 16); g = int(hs[2:4], 16); b = int(hs[4:6], 16)
                                palette_bgr.append((b, g, r))
                        except Exception:
                            continue
            except Exception:
                palette_bgr = []
            if not palette_bgr:
                # Hard fallback to known defect palette
                for hs in ["#FCFF8A", "#7FD47F", "#ECA360", "#6AD0FF", "#4A4A4A"]:
                    try:
                        hs = hs.lstrip("#")
                        if len(hs) == 6:
                            r = int(hs[0:2], 16); g = int(hs[2:4], 16); b = int(hs[4:6], 16)
                            palette_bgr.append((b, g, r))
                    except Exception:
                        continue
            palette_fallback = palette_bgr[0] if palette_bgr else (255, 200, 0)
            dets = []
            try:
                dets = solvision_manager.detect_for('defect', str(bbox_path), score_threshold=override_thr)
            except Exception as ex:
                self.tt_message.emit(f"[Step4] idx {idx}: detect failed: {ex}")
                dets = []

            ann = img.copy()
            if not dets:
                # No detections; still use palette color instead of red
                _cv2.putText(ann, 'No defects', (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, palette_fallback, 2)
            else:
                for det in dets:
                    b = det.get('bounds')
                    if not b or len(b) < 4:
                        continue
                    x, y, w, h = b
                    try:
                        x = int(round(float(x))); y = int(round(float(y)))
                        w = int(round(float(w))); h = int(round(float(h)))
                    except Exception:
                        continue
                    try:
                        cid = det.get("class_id")
                        idx = int(cid) if cid is not None else 0
                    except Exception:
                        idx = 0
                    if idx < 0 or idx >= len(palette_bgr):
                        idx = 0
                    color = palette_bgr[idx] if palette_bgr else palette_fallback
                    _cv2.rectangle(ann, (x, y), (x + w, y + h), color, 2)
                    label = str(det.get('class') or 'defect')
                    try:
                        sc = det.get('score')
                        if sc is not None:
                            label = f"{label} {float(sc):.2f}"
                    except Exception:
                        pass
                    if label:
                        _cv2.putText(ann, label, (x + 4, max(0, y - 6)), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out_ann = str(step4_dir / f"step-04_defect_{idx:03d}.png")
            _cv2.imwrite(out_ann, ann)
            self.tt_message.emit(f"[Step4] idx {idx}: saved {out_ann}")
        except Exception as ex:
            try:
                self.tt_message.emit(f"[Step4] idx {idx}: failed: {ex}")
            except Exception:
                pass
    # ---- Step 3: run front-attachment detectron on step-02 crops ----
    def _run_step3_front(self, step2_dir):
        from pathlib import Path as _Path
        import re as _re
        import cv2 as _cv2
        import numpy as _np
        from services.config import state as _state
        from services import solvision_manager

        step2_dir = _Path(step2_dir)
        crops_dir = step2_dir / 'step_2_cropped'
        if not crops_dir.exists():
            self.tt_message.emit("[Step3] No step-02 crops found; skipping.")
            return

        # Require front model already loaded by user
        front_path = solvision_manager.current_project_path_for('front')
        if not front_path:
            self.tt_message.emit("[Step3] Front model not loaded; please load it before running Step 3.")
            return

        step3_dir = step2_dir.parent / 'step-03'
        try:
            step3_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        files = sorted([p for p in crops_dir.glob('step-02_front_crop_*.png')])
        rx = _re.compile(r"step-02_front_crop_(\d+)\.png$", _re.IGNORECASE)
        total = 0
        for p in files:
            m = rx.search(p.name)
            if not m:
                continue
            idx = int(m.group(1))
            try:
                img = _cv2.imread(str(p))
                if img is None:
                    self.tt_message.emit(f"[Step3] idx {idx}: failed to read {p}")
                    continue
                dets = solvision_manager.detect_for('front', str(p))
                H, W = img.shape[:2]
                if not dets:
                    # Save an annotated image with note
                    ann = img.copy()
                    _cv2.putText(ann, 'No detection', (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    out_ann = str(step3_dir / f"step-03_front_{idx:03d}.png")
                    _cv2.imwrite(out_ann, ann)
                    self.tt_message.emit(f"[Step3] idx {idx}: no detection; saved {out_ann}")
                    total += 1
                    continue
                # Choose detection closest to image center (tie-break by higher score)
                cx0, cy0 = W * 0.5, H * 0.5
                def _metric(d):
                    try:
                        b = d.get('bounds') or d.get('rect') or None
                        if not b or len(b) < 4:
                            return (float('inf'), -0.0)
                        x, y, w, h = b
                        x = float(x); y = float(y); w = float(w); h = float(h)
                        cx = x + w * 0.5; cy = y + h * 0.5
                        dist2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
                        sc = float(d.get('score') or 0.0)
                        return (dist2, -sc)
                    except Exception:
                        return (float('inf'), -0.0)
                best = min(dets, key=_metric)
                bx, by, bw, bh = best.get('bounds') or (0, 0, 0, 0)
                try:
                    bx = int(round(float(bx))); by = int(round(float(by)))
                    bw = int(round(float(bw))); bh = int(round(float(bh)))
                except Exception:
                    bx, by, bw, bh = 0, 0, 0, 0
                # Clamp
                bx = max(0, min(W - 1, bx)); by = max(0, min(H - 1, by))
                bw = max(0, min(W - bx, bw)); bh = max(0, min(H - by, bh))

                ann = img.copy()
                def _safe_label_pos(x, y, w, h, text):
                    # Clamp horizontally; prefer above box, else below while keeping inside image.
                    (tw, th), _ = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    lx = max(0, min(W - tw - 1, x + 4))
                    if y - th - 6 >= 0:
                        ly = y - 6
                    else:
                        ly = min(H - 6, y + h + th)
                    ly = max(th, min(H - 1, ly))
                    return lx, ly

                def _color_from_meta(det_obj, role='front'):
                    try:
                        cid = det_obj.get("class_id")
                        colors = solvision_manager.class_colors_for(role)
                        if colors and cid is not None and 0 <= int(cid) < len(colors):
                            hs = colors[int(cid)].lstrip("#")
                            if len(hs) == 6:
                                r = int(hs[0:2], 16); g = int(hs[2:4], 16); b = int(hs[4:6], 16)
                                return (b, g, r)
                    except Exception:
                        pass
                    return (0, 255, 0)

                color = _color_from_meta(best, 'front')
                _cv2.rectangle(ann, (bx, by), (bx + bw, by + bh), color, 2)
                label = str(best.get('class') or '')
                try:
                    sc = best.get('score')
                    if sc is not None:
                        label = f"{label} {float(sc):.2f}" if label else f"{float(sc):.2f}"
                except Exception:
                    pass
                if label:
                    lx, ly = _safe_label_pos(bx, by, bw, bh, label)
                    _cv2.putText(ann, label, (lx, ly), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                out_ann = str(step3_dir / f"step-03_front_{idx:03d}.png")
                _cv2.imwrite(out_ann, ann)

                # Save bbox crop
                pad = 50
                x0 = max(0, bx - pad)
                y0 = max(0, by - pad)
                x1 = min(W, bx + bw + pad)
                y1 = min(H, by + bh + pad)
                crop = img[y0:y1, x0:x1].copy() if (x1 > x0 and y1 > y0) else img.copy()
                out_crop = str(step3_dir / f"step-03_front_bbox_{idx:03d}.png")
                _cv2.imwrite(out_crop, crop)
                self.tt_message.emit(f"[Step3] idx {idx}: saved {out_ann} and bbox {out_crop}")
                total += 1
            except Exception as ex:
                self.tt_message.emit(f"[Step3] idx {idx}: failed: {ex}")
        self.tt_message.emit(f"[Step3] Done. Processed {total} cropped image(s)")

    # ---- Step 4: run defect model on Step 3 bbox crops ----
    def _run_step4_defect(self, step2_dir):
        from pathlib import Path as _Path
        import re as _re
        import cv2 as _cv2
        import numpy as _np
        from services.config import state as _state
        from services import solvision_manager

        step2_dir = _Path(step2_dir)
        step3_dir = step2_dir.parent / 'step-03'
        step4_dir = step2_dir.parent / 'step-04'
        bbox_files = sorted(step3_dir.glob('step-03_front_bbox_*.png'))
        if not bbox_files:
            self.tt_message.emit("[Step4] No Step-03 bbox crops found; skipping.")
            return

        from services.config import state as _state
        st = _state()
        defect_thr = getattr(st, 'defect_score_threshold', None)
        defect_loaded_path = solvision_manager.current_project_path_for('defect')
        if not defect_loaded_path:
            self.tt_message.emit("[Step4] Defect model not loaded; please load it before running Step 4.")
            return

        try:
            step4_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        def _hex_to_bgr(hs):
            try:
                hs = str(hs).lstrip("#").strip()
                if len(hs) == 6:
                    r = int(hs[0:2], 16); g = int(hs[2:4], 16); b = int(hs[4:6], 16)
                    return (b, g, r)
            except Exception:
                pass
            return None

        # Precompute palette from defect model metadata only (no fallbacks).
        palette_bgr = []
        try:
            cols = solvision_manager.class_colors_for('defect')
            if cols:
                for hs in cols:
                    c = _hex_to_bgr(hs)
                    if c is not None:
                        palette_bgr.append(c)
        except Exception:
            palette_bgr = []

        def _color_for_det(det_obj):
            # Prefer the exact color coming from Detectron metadata; otherwise map by class id if available.
            c = _hex_to_bgr(det_obj.get('color'))
            if c is not None:
                return c
            try:
                cid = det_obj.get("class_id")
                if palette_bgr and cid is not None and 0 <= int(cid) < len(palette_bgr):
                    return palette_bgr[int(cid)]
            except Exception:
                pass
            return None

        rx = _re.compile(r"step-03_front_bbox_(\d+)\.png$", _re.IGNORECASE)
        total = 0
        for p in bbox_files:
            m = rx.search(p.name)
            if not m:
                continue
            idx = int(m.group(1))
            try:
                img = _cv2.imread(str(p))
                if img is None:
                    self.tt_message.emit(f"[Step4] idx {idx}: failed to read {p}")
                    continue
                H, W = img.shape[:2]
                dets = solvision_manager.detect_for('defect', str(p), score_threshold=defect_thr)
                ann = img.copy()
                if not dets:
                    if palette_bgr:
                        _cv2.putText(ann, 'No defects', (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, palette_bgr[0], 2)
                else:
                    def _label_pos(x, y, w, h, text):
                        # Keep text fully inside image: clamp horizontally and flip below when needed.
                        (tw, th), _ = _cv2.getTextSize(text, _cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        lx = max(0, min(W - tw - 1, x + 4))
                        if y - th - 6 >= 0:
                            ly = y - 6
                        else:
                            ly = min(H - 6, y + h + th)
                        ly = max(th, min(H - 1, ly))
                        return lx, ly
                    for det in dets:
                        b = det.get('bounds')
                        if not b or len(b) < 4:
                            continue
                        x, y, w, h = b
                        try:
                            x = int(round(float(x))); y = int(round(float(y)))
                            w = int(round(float(w))); h = int(round(float(h)))
                        except Exception:
                            continue
                        color = _color_for_det(det)
                        if color is None:
                            continue
                        _cv2.rectangle(ann, (x, y), (x + w, y + h), color, 2)
                        label = str(det.get('class') or 'defect')
                        try:
                            sc = det.get('score')
                            if sc is not None:
                                label = f"{label} {float(sc):.2f}"
                        except Exception:
                            pass
                        if label:
                            lx, ly = _label_pos(x, y, w, h, label)
                            _cv2.putText(ann, label, (lx, ly), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                out_ann = str(step4_dir / f"step-04_defect_{idx:03d}.png")
                _cv2.imwrite(out_ann, ann)
                self.tt_message.emit(f"[Step4] idx {idx}: saved {out_ann}")
                total += 1
            except Exception as ex:
                self.tt_message.emit(f"[Step4] idx {idx}: failed: {ex}")
        self.tt_message.emit(f"[Step4] Done. Processed {total} bbox crop(s)")

    # Camera slots
    def on_camera_refresh(self):
        try:
            devices = camera_service.enumerate_devices()
            self.workflow_tab.camera_panel.set_devices(devices)
            backend = camera_service.backend_name()
            self.workflow_tab.append_log(f"[Camera] Backend(s): {backend} | Found {len(devices)} device(s).")
            if len(devices) == 0:
                diag = camera_service.diagnostics()
                # Log iRAYPLE diagnostics when present (for SDK path issues)
                iray = diag.get("iRAYPLE") or {}
                py_dir = iray.get("py_dir")
                rt_dir = iray.get("runtime_dir")
                enum_ret = iray.get("enum_ret")
                dev_num = iray.get("dev_num")
                version = iray.get("version")
                import_ok = iray.get("import_ok")
                load_error = iray.get("load_error")
                if version:
                    self.workflow_tab.append_log(f"[Camera] iRAYPLE SDK version: {version}")
                if py_dir or rt_dir:
                    self.workflow_tab.append_log(f"[Camera] iRAYPLE paths: PY={py_dir} RT={rt_dir}")
                if enum_ret is not None or dev_num is not None:
                    self.workflow_tab.append_log(f"[Camera] iRAYPLE enum result: ret={enum_ret} dev_num={dev_num}")
                if import_ok is not None:
                    self.workflow_tab.append_log(f"[Camera] iRAYPLE import_ok={import_ok}")
                if load_error:
                    try:
                        last_line = str(load_error).splitlines()[-1]
                    except Exception:
                        last_line = str(load_error)
                    self.workflow_tab.append_log(f"[Camera] iRAYPLE load_error: {last_line}")
        except Exception as ex:
            self.workflow_tab.append_log(f"[Camera] Enumeration failed: {ex}")

    def _device_name(self, index: int) -> str:
        # Find name from the current panel list
        for widget in (self.workflow_tab.camera_panel.top, self.workflow_tab.camera_panel.front):
            for i in range(widget.selector.count()):
                data = widget.selector.itemData(i)
                if isinstance(data, dict) and data.get("index") == index:
                    return widget.selector.itemText(i)
        return f"Camera {index}"

    def on_camera_connect(self, role: str, index: int):
        # prevent same device for both roles
        other_role = "Front" if role == "Top" else "Top"
        if camera_service.get_connected_index(other_role) == index:
            self.workflow_tab.append_log(
                f"[Camera] Selected device already assigned to {other_role.lower()} camera."
            )
            return
        ok = camera_service.connect(role, index)
        if ok:
            name = self._device_name(index)
            self.workflow_tab.camera_panel.set_connected(role, True, name)
            self.workflow_tab.append_log(f"[Camera] {role} connected to {name}.")
            # Persist selection
            st = state()
            if role == "Top":
                st.camera_top_index = index
            else:
                st.camera_front_index = index
            save_state()
            # Auto-capture on connect
            self.on_camera_capture(role)
        else:
            self.workflow_tab.append_log(f"[Camera] {role} connection failed for index {index}.")

    def on_camera_disconnect(self, role: str):
        camera_service.disconnect(role)
        self.workflow_tab.camera_panel.set_connected(role, False)
        self.workflow_tab.append_log(f"[Camera] {role} disconnected.")

    def on_camera_capture(self, role: str):
        try:
            frame = cammgr.capture(role)
            pm = np_bgr_to_qpixmap(frame)
            if role == "Top":
                self.preview_panel.set_original_np(pm)
            else:
                self.preview_panel.set_front_np(pm)
            self.workflow_tab.append_log(f"[Camera] Captured preview from {role.lower()} camera.")
        except Exception as ex:
            self.workflow_tab.append_log(f"[Camera] Capture failed on {role}: {ex}")

    def on_camera_selected(self, role: str, index: int):
        # Save selection without auto-connecting
        st = state()
        if role == "Top":
            st.camera_top_index = index
        else:
            st.camera_front_index = index
        save_state()

    # Selected Files handlers (main screen group)
    def on_load_attachment_file(self, path: str):
        if not path:
            return
        # Load attachment model as the active project without blocking the UI
        try:
            self.workflow_tab.append_log(f"[Detectron] Loading attachment model: {path}")
            # Optimistically mark as loaded so the button turns green immediately
            try:
                self.workflow_tab.set_attachment_loaded(True)
            except Exception:
                pass
        except Exception:
            pass

        import threading

        def _load():
            from PyQt5.QtCore import QTimer
            try:
                project_loader.load_project(path)
                try:
                    st = state(); st.attachment_path = path; st.last_project_path = path; save_state()
                except Exception:
                    pass
                # Update UI on the main thread
                try:
                    QTimer.singleShot(0, lambda: self.workflow_tab.set_attachment_loaded(True))
                except Exception:
                    pass
                self.tt_message.emit(f"[Detectron] Attachment model loaded: {path}")
            except Exception as ex:
                err_msg = f"Failed to load project.\n{ex}"
                try:
                    from services import solvision_manager
                    diag = solvision_manager.diagnostics_text()
                    self.tt_message.emit("[Detectron] Init diagnostics:\n" + diag)
                except Exception:
                    pass
                try:
                    QTimer.singleShot(0, lambda: self.workflow_tab.set_attachment_loaded(False))
                except Exception:
                    pass
                try:
                    QTimer.singleShot(0, lambda: QMessageBox.warning(self, "Load Attachment", err_msg))
                except Exception:
                    pass

        threading.Thread(target=_load, daemon=True).start()

    def on_load_front_file(self, path: str):
        if not path:
            return
        # Persist for future steps; style button as loaded
        try:
            st = state(); st.front_attachment_path = path; save_state()
            try:
                self.workflow_tab.set_front_loaded(True)
            except Exception:
                pass
            self.workflow_tab.append_log(f"[Detectron] Front model set: {path}")
            # Preload into its own session asynchronously to avoid blocking the UI
            try:
                import threading
                from services import solvision_manager
                def _load():
                    try:
                        solvision_manager.load_project_for('front', path, mode='exe')
                        self.tt_message.emit("[Detectron] Front model loaded in dedicated session.")
                    except Exception as ex:
                        self.tt_message.emit(f"[Detectron] Front model load failed: {ex}")
                threading.Thread(target=_load, daemon=True).start()
            except Exception:
                pass
        except Exception:
            pass

    def on_load_defect_file(self, path: str):
        if not path:
            return
        # Persist for future steps; style button as loaded
        try:
            st = state(); st.defect_path = path; save_state()
            try:
                self.workflow_tab.set_defect_loaded(True)
            except Exception:
                pass
            self.workflow_tab.append_log(f"[Detectron] Defect model set: {path}")
            # Preload defect model in its own session asynchronously
            try:
                import threading
                from services import solvision_manager
                def _load():
                    try:
                        solvision_manager.load_project_for('defect', path, mode='exe')
                        self.tt_message.emit("[Detectron] Defect model loaded in dedicated session.")
                    except Exception as ex:
                        self.tt_message.emit(f"[Detectron] Defect model load failed: {ex}")
                threading.Thread(target=_load, daemon=True).start()
            except Exception:
                pass
        except Exception:
            pass

    def closeEvent(self, event):
        try:
            camera_service.release_all()
            if hasattr(self, "_tt_listener") and self._tt_listener:
                turntable_service.remove_listener(self._tt_listener)
            turntable_service.disconnect()
            try:
                solvision_manager.dispose()
            except Exception:
                pass
        finally:
            super().closeEvent(event)
