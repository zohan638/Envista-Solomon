from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import concurrent.futures
import cv2
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QFileDialog, QMessageBox

from controllers.hardware_controller import HardwareController
from controllers.live_camera_controller import LiveCameraController
from controllers.model_controller import ModelController
from models.run_context import RunContext
from services import camera_manager as cammgr
from services import camera_service, contour_tools, linear_axis_service, solvision_manager, debug_log
from services.app_paths import app_root
from services.config import save_state, settings, state
from ui.edge_tuner import EdgeTunerDialog
from ui.qt_image import np_bgr_to_qpixmap

# Horizontal FOV of the front camera when measured inside the top-camera image (pixels)
DEFAULT_FRONT_FOV_TOP_PX = 441.4
# Front-view calibration: 1450 actuator steps corresponds to 1270 px in the front image.
FRONT_STEPS_PER_PIXEL = 1450.0 / 1270.0
FRONT_IMAGE_WIDTH_PX = 2464.0


class WorkflowController(QObject):
    """
    Coordinates the 4-step inspection workflow.

    The controller owns calls into services (models/hardware/cameras) and
    updates the view via the widgets it is handed, leaving Qt widgets focused
    on presentation.
    """

    def __init__(
        self,
        view,
        live_controller: LiveCameraController,
        hardware_controller: HardwareController,
        model_controller: ModelController,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.view = view
        self.workflow_tab = view.workflow_tab
        self.preview_panel = view.preview_panel
        self.defect_ledger = view.defect_ledger
        self.tt_message = view.tt_message
        self.tt_status = view.tt_status
        self.plc_snapshot = view.plc_snapshot
        self.live = live_controller
        self.hardware = hardware_controller
        self.models = model_controller

        self._current_image_path: Optional[str] = None
        self._last_top_detections: List[dict] = []
        self._attachment_defect_state: Dict[object, str] = {}
        self._top_raw_np = None
        self._cycle_start_ts: Optional[float] = None
        self._cycle_cap_dir: Optional[Path] = None
        self._last_capture_path: Optional[str] = None
        self._front_results: Dict[int, str] = {}
        self._attachment_count: int = 0
        self._last_cap_root = None
        self._suppress_ledger_events = False

    def _normalize_cap_root(self, cap_dir):
        """Return the capture root folder (parent of step-0X if such a folder was passed)."""
        if cap_dir is None:
            return None
        root = Path(cap_dir)
        if root.name in {"step-01", "step-02", "step-03", "step-04"}:
            root = root.parent
        return root

    # ----- Attachment overlay bookkeeping -----
    def _set_top_detections(self, dets):
        try:
            self._last_top_detections = [dict(d) for d in (dets or [])]
        except Exception:
            self._last_top_detections = list(dets or [])
        try:
            self.preview_panel.set_attachment_detections(self._last_top_detections)
        except Exception:
            pass
        self._update_top_annotation()

    def _apply_defect_states_to_overlay(self):
        try:
            dets = []
            for d in self._last_top_detections:
                try:
                    nd = dict(d)
                except Exception:
                    nd = d
                try:
                    idx_val = int(nd.get("index", None))
                except Exception:
                    idx_val = nd.get("index")
                st = self._attachment_defect_state.get(idx_val)
                if st is not None:
                    nd["defect_state"] = st
                dets.append(nd)
            base_pm = None
            try:
                if self._top_raw_np is not None:
                    base_pm = np_bgr_to_qpixmap(self._top_raw_np)
            except Exception:
                base_pm = None
            composed = None
            if base_pm is not None:
                composed = self.preview_panel.render_attachment_overlay(base_pm, dets)
            if composed is not None and (not composed.isNull()):
                try:
                    self.defect_ledger.set_top_pixmap(
                        composed,
                        detections=dets,
                        image_size=(composed.width(), composed.height()),
                    )
                except Exception:
                    self.defect_ledger.set_top_pixmap(composed)
        except Exception:
            pass

    def _update_top_annotation(self):
        try:
            if self._top_raw_np is None:
                if self._last_capture_path and os.path.exists(self._last_capture_path):
                    self._top_raw_np = cv2.imread(self._last_capture_path)
        except Exception:
            pass
        self._apply_defect_states_to_overlay()

    def _set_defect_state(self, idx, state: str):
        try:
            key = int(idx)
        except Exception:
            key = idx
        self._attachment_defect_state[key] = state
        try:
            QTimer.singleShot(0, self._update_top_annotation)
        except Exception:
            self._update_top_annotation()

    def _annotate_top_detections(self, img_path: Optional[str], detections):
        """Compute arrows, indices, phi, and offsets for top-view detections (Step 1)."""
        if not detections:
            return detections
        try:
            import cv2 as _cv2
            import math

            src = None
            if self._top_raw_np is not None:
                src = self._top_raw_np
            elif img_path and os.path.exists(img_path):
                src = _cv2.imread(img_path)
            if src is None:
                return detections

            try:
                from services.config import state as _state

                params = getattr(_state(), "contour_params", None)
            except Exception:
                params = None

            try:
                arrows, _ = contour_tools.compute_arrows_for_detections(src, detections, params=params)
            except Exception as ex:
                arrows = [{} for _ in detections]
                try:
                    self.workflow_tab.append_log(f"[Detectron] Arrow computation skipped: {ex}")
                except Exception:
                    pass

            try:
                for det, arr in zip(detections, arrows):
                    if isinstance(det, dict) and isinstance(arr, dict):
                        det["arrow"] = arr
            except Exception:
                pass

            h, w = src.shape[:2]
            cx0, cy0 = w / 2.0, h / 2.0
            pts = []
            for det in detections:
                if not isinstance(det, dict):
                    continue
                b = det.get("bounds")
                if not b:
                    continue
                x, y, bw, bh = b
                cx = x + bw / 2.0
                cy = y + bh / 2.0
                det["det_center"] = (cx, cy)
                det["center"] = (cx0, cy0)
                det["center_ref"] = (cx0, cy0)
                det["offset_top_px"] = cx - cx0
                ang = math.atan2(cy0 - cy, cx - cx0)
                pts.append((cx, cy, ang, det))

            if pts:
                ref = -math.pi * 0.25

                def _ang_diff(a, b=ref):
                    r = a - b
                    while r <= -math.pi:
                        r += 2 * math.pi
                    while r > math.pi:
                        r -= 2 * math.pi
                    return abs(r)

                start = min(pts, key=lambda t: _ang_diff(t[2]))
                ang0 = start[2]

                def _rank(p):
                    r = p[2] - ang0
                    while r < 0:
                        r += 2 * math.pi
                    while r >= 2 * math.pi:
                        r -= 2 * math.pi
                    return r

                pts.sort(key=_rank)
                for i, (_, _, _, det) in enumerate(pts, start=1):
                    try:
                        det["index"] = i
                    except Exception:
                        pass

            for det in detections:
                if not isinstance(det, dict):
                    continue
                vec = det.get("arrow", {}).get("vec")
                if vec is None:
                    continue
                try:
                    vx, vy = float(vec[0]), float(vec[1])
                except Exception:
                    continue
                phi = -math.atan2(vx, -vy)
                det["phi"] = phi
                cx, cy = det.get("det_center") or (None, None)
                cx0, cy0 = det.get("center_ref") or (None, None)
                if cx is not None and cx0 is not None:
                    dx = float(cx - cx0)
                    dy = float(cy - cy0)
                    x_rot_off = dx * math.cos(phi) - dy * math.sin(phi)
                    det["offset_top_rot_px"] = x_rot_off
                    try:
                        idx_val = det.get("index")
                        self.workflow_tab.append_log(
                            f"[Step1] idx={idx_val} det_center=({cx:.1f},{cy:.1f}) img_center=({cx0:.1f},{cy0:.1f}) "
                            f"phi={math.degrees(phi):.2f}deg off_top_px={dx:.1f} off_top_rot_px={x_rot_off:.1f}"
                        )
                    except Exception:
                        pass
            return detections
        except Exception as ex:
            try:
                self.workflow_tab.append_log(f"[Detectron] Top metadata skipped: {ex}")
            except Exception:
                pass
            return detections

    # ----- UI commands -----
    def load_image(self):
        try:
            self.live.stop_live()
        except Exception:
            pass
        path, _ = QFileDialog.getOpenFileName(
            self.view,
            "Choose Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)",
        )
        if not path:
            self.live.start_live()
            return
        self._current_image_path = path
        try:
            st = state()
            st.last_image_path = path
            save_state()
        except Exception:
            pass
        self.workflow_tab.append_log(f"Loaded image: {path}")
        self.preview_panel.set_original_image(path)
        self.preview_panel.set_front_preview_image(path)
        self.live.start_live()

    def open_tuner(self):
        try:
            self.live.stop_live()
        except Exception:
            pass
        dlg = EdgeTunerDialog(self.view)
        dlg.exec_()
        try:
            st = state()
            st.contour_params = dlg.current_params()
            save_state()
        except Exception:
            pass
        try:
            QTimer.singleShot(0, self.live.start_live)
        except Exception:
            pass

    def on_prev(self):
        try:
            self.defect_ledger.select_prev()
        except Exception:
            pass

    def on_next(self):
        try:
            self.defect_ledger.select_next()
        except Exception:
            pass

    def on_attachment_clicked(self, idx):
        try:
            self.defect_ledger.set_current_index(idx)
        except Exception:
            pass

    def on_defect_threshold_changed(self, v: float):
        try:
            self.tt_message.emit(f"[Step4] Defect threshold override: {float(v):.3f}")
        except Exception:
            pass

    def _invoke_ui(self, fn):
        try:
            QTimer.singleShot(0, fn)
        except Exception:
            try:
                fn()
            except Exception:
                pass
        try:
            debug_log.log("ui_invoke scheduled")
        except Exception:
            pass

    def _with_ledger_suppressed(self, fn):
        if self._suppress_ledger_events:
            try:
                return fn()
            except Exception:
                return None
        self._suppress_ledger_events = True
        try:
            return fn()
        finally:
            self._suppress_ledger_events = False

    def _update_ledger_front(self, idx: int, path: Optional[str], message: Optional[str] = None):
        def _ui():
            pm = None
            msg = message
            try:
                if path:
                    import os as _os

                    if _os.path.exists(path):
                        pm = QPixmap(path)
                    else:
                        msg = msg or "File missing."
            except Exception:
                pm = None
            try:
                self._with_ledger_suppressed(lambda: self.defect_ledger.set_front_for_index(idx, pm, message=msg))
            except Exception:
                pass

        self._invoke_ui(_ui)

    def _current_cap_root(self):
        try:
            if self._last_cap_root:
                return self._last_cap_root
        except Exception:
            pass
        try:
            if self._cycle_cap_dir:
                return self._normalize_cap_root(self._cycle_cap_dir)
        except Exception:
            pass
        try:
            if self._last_capture_path:
                from pathlib import Path as _Path

                return self._normalize_cap_root(_Path(self._last_capture_path).parent)
        except Exception:
            pass
        return None

    def _maybe_load_front_from_disk(self, idx):
        cap_root = self._current_cap_root()
        if cap_root is None:
            return
        from pathlib import Path
        try:
            cached = self._front_results.get(int(idx))
            if cached:
                p_cached = Path(cached)
                if p_cached.exists():
                    self._update_ledger_front(idx, str(p_cached))
                    return
        except Exception:
            pass
        target = Path(cap_root) / "step-04" / f"step-04_defect_{int(idx):03d}.png"
        try:
            if target.exists():
                self._front_results[int(idx)] = str(target)
                self._update_ledger_front(idx, str(target))
                return
        except Exception:
            pass
        try:
            self._update_ledger_front(idx, None, "No step-4 image yet.")
        except Exception:
            pass

    def on_ledger_selection_changed(self, idx):
        if self._suppress_ledger_events:
            return
        try:
            self._maybe_load_front_from_disk(idx)
        except Exception:
            pass

    # ----- Model loading -----
    def load_attachment_file(self, path: str):
        if not path:
            return
        self.workflow_tab.append_log(f"[Detectron] Loading attachment model: {path}")
        self.workflow_tab.set_attachment_loaded(True)
        debug_log.log(f"ui_request load_attachment path={path}")

        def _on_success(_path: str):
            self._invoke_ui(
                lambda: (
                    self.workflow_tab.set_attachment_loaded(True),
                    self.tt_message.emit(f"[Detectron] Attachment model loaded: {path}"),
                    self.workflow_tab.set_selected_files(attachment=path),
                )
            )

        def _on_error(_path: str, err: str):
            self._invoke_ui(
                lambda: (
                    self.workflow_tab.set_attachment_loaded(False),
                    QMessageBox.warning(self.view, "Load Attachment", f"Failed to load project.\n{err}"),
                )
            )

        self.models.load_attachment(path, on_success=_on_success, on_error=_on_error)

    def load_front_file(self, path: str):
        if not path:
            return
        try:
            st = state()
            st.front_attachment_path = path
            save_state()
        except Exception:
            pass
        self.workflow_tab.set_front_loaded(True)
        self.workflow_tab.append_log(f"[Detectron] Front model set: {path}")

        def _on_success(_path: str):
            def _ui():
                self.tt_message.emit("[Detectron] Front model loaded in dedicated session.")
                self.workflow_tab.set_selected_files(front=path)
                st = state()
                st.front_attachment_path = path
                st.last_front_project_path = path
                save_state()

            self._invoke_ui(_ui)

        def _on_error(_path: str, err: str):
            self._invoke_ui(lambda: self.tt_message.emit(f"[Detectron] Front model load failed: {err}"))

        self.models.load_front(path, on_success=_on_success, on_error=_on_error)

    def load_defect_file(self, path: str):
        if not path:
            return
        try:
            st = state()
            st.defect_path = path
            save_state()
        except Exception:
            pass
        self.workflow_tab.set_defect_loaded(True)
        self.workflow_tab.append_log(f"[Detectron] Defect model set: {path}")

        def _on_success(_path: str):
            def _ui():
                self.tt_message.emit("[Detectron] Defect model loaded in dedicated session.")
                self.workflow_tab.set_selected_files(defect=path)
                st = state()
                st.defect_path = path
                st.last_defect_project_path = path
                save_state()

            self._invoke_ui(_ui)

        def _on_error(_path: str, err: str):
            self._invoke_ui(lambda: self.tt_message.emit(f"[Detectron] Defect model load failed: {err}"))

        self.models.load_defect(path, on_success=_on_success, on_error=_on_error)

    # ----- Workflow steps -----
    def _build_run_context(self, part_id_raw: str) -> Optional[RunContext]:
        from datetime import datetime
        import re

        if not part_id_raw:
            QMessageBox.information(self.view, "Run Detection", "Please enter a Part ID before running detection.")
            return None
        part_id_clean = re.sub(r"[^A-Za-z0-9._-]+", "_", part_id_raw).strip("_")
        if not part_id_clean:
            part_id_clean = "part"
        try:
            st = state()
            st.part_id = part_id_raw
            save_state()
        except Exception:
            pass
        try:
            self._cycle_start_ts = time.time()
        except Exception:
            self._cycle_start_ts = None

        cap_dir = None
        try:
            ts = datetime.now()
            base = app_root() / "captures" / part_id_clean
            cap_dir = base / ts.strftime("%Y-%m-%d") / ts.strftime("%H%M%S")
            cap_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            cap_dir = None
        try:
            self._cycle_cap_dir = cap_dir
            self._last_cap_root = self._normalize_cap_root(cap_dir)
        except Exception:
            pass
        return RunContext(part_id_raw=part_id_raw, part_id_clean=part_id_clean, capture_dir=cap_dir, cycle_start_ts=self._cycle_start_ts or 0.0)

    def run_detection(self):
        try:
            self.live.stop_live()
        except Exception:
            pass
        try:
            self._attachment_defect_state = {}
            self._top_raw_np = None
            self._set_top_detections([])
        except Exception:
            pass
        try:
            self._front_results = {}
            self._attachment_count = 0
            self.defect_ledger.clear_items(total=0)
            self._last_cap_root = None
        except Exception:
            pass
        try:
            QTimer.singleShot(0, self.live.start_live)
        except Exception:
            pass

        ctx = self._build_run_context(self.workflow_tab.part_id())
        if ctx is None:
            return
        try:
            self.workflow_tab.append_log(f"[Capture] Part ID set to '{ctx.part_id_raw}' (folder '{ctx.part_id_clean}')")
        except Exception:
            pass

        # Offer to open tuner first if no saved contour params yet
        try:
            st = state()
            if getattr(st, "contour_params", None) is None:
                resp = QMessageBox.question(
                    self.view,
                    "Edge/Contour Tuner",
                    "Open tuner to calibrate edge/contour before detection?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if resp == QMessageBox.Yes:
                    self.open_tuner()
        except Exception:
            pass

        if not self.models.ensure_models_loaded(
            required=("top", "front", "defect"),
            parent=self.view,
            show_dialog=True,
            log_callback=self.workflow_tab.append_log,
        ):
            return

        img_path = None
        capture_error = None
        try:
            if camera_service.is_connected("Top"):
                frame = cammgr.capture("Top")
                pm = np_bgr_to_qpixmap(frame)
                self.preview_panel.set_original_np(pm)
                try:
                    import numpy as _np

                    self._top_raw_np = _np.array(frame).copy()
                except Exception:
                    self._top_raw_np = None
                if ctx.capture_dir is not None:
                    try:
                        raw_path = str((ctx.capture_dir / "step-01_top_raw.png"))
                        cv2.imwrite(raw_path, frame)
                        self.workflow_tab.append_log(f"[Capture] Saved raw: {raw_path}")
                        img_path = raw_path
                    except Exception:
                        pass
                if not img_path:
                    try:
                        tmp = os.path.join(os.getcwd(), "top_capture.png")
                        cv2.imwrite(tmp, frame)
                        img_path = tmp
                    except Exception:
                        pass
            else:
                img_path = self._current_image_path
                if ctx.capture_dir is not None and img_path:
                    try:
                        import shutil

                        if os.path.exists(img_path):
                            dst = str((ctx.capture_dir / "step-01_attachment_source.png"))
                            shutil.copyfile(img_path, dst)
                            self.workflow_tab.append_log(f"[Capture] Copied source: {dst}")
                            img_path = dst
                    except Exception:
                        pass
                try:
                    if img_path and os.path.exists(img_path):
                        self._top_raw_np = cv2.imread(img_path)
                except Exception:
                    pass
        except Exception as ex:
            capture_error = str(ex)
            self.workflow_tab.append_log(f"[Camera] Capture failed: {capture_error}")
            img_path = self._current_image_path

        if not img_path:
            if capture_error:
                QMessageBox.information(
                    self.view,
                    "Run Detection",
                    "Top camera capture failed:\n"
                    f"{capture_error}\n\n"
                    "Please reconnect the top camera or load an image first.",
                )
            else:
                QMessageBox.information(
                    self.view,
                    "Run Detection",
                    "No image available. Connect the top camera or load an image first.",
                )
            return

        try:
            if ctx.capture_dir:
                self._last_capture_path = str(ctx.capture_dir / "step-01_top_raw.png")
            else:
                self._last_capture_path = img_path
        except Exception:
            self._last_capture_path = img_path

        # Step 1 - attachment detection
        dets = []
        y_offset = 0
        try:
            det_source = img_path
            y0, y1 = 530, 2030
            roi_path = None
            if self._top_raw_np is None and img_path and os.path.exists(img_path):
                try:
                    self._top_raw_np = cv2.imread(img_path)
                except Exception:
                    self._top_raw_np = None
            if self._top_raw_np is not None:
                h = self._top_raw_np.shape[0]
                y_start = max(0, min(h, int(y0)))
                y_end = max(y_start, min(h, int(y1)))
                if y_end > y_start:
                    roi = self._top_raw_np[y_start:y_end, :]
                    y_offset = y_start
                    try:
                        if ctx.capture_dir is not None:
                            roi_path = str((ctx.capture_dir / "step-01_top_raw_roi.png"))
                        else:
                            roi_path = os.path.join(os.getcwd(), "top_capture_roi.png")
                        cv2.imwrite(roi_path, roi)
                        det_source = roi_path
                        try:
                            self.workflow_tab.append_log(f"[Detectron] Using ROI for Step1: y={y_start}:{y_end}")
                        except Exception:
                            pass
                    except Exception:
                        det_source = img_path
                        roi_path = None
            dets = solvision_manager.detect(det_source)
            if y_offset and dets:
                for d in dets:
                    b = d.get("bounds")
                    if not b:
                        continue
                    try:
                        x, y, w, h = b
                        d["bounds"] = (x, y + y_offset, w, h)
                    except Exception:
                        pass
        except Exception as ex:
            QMessageBox.information(self.view, "Run Detection", f"Detection failed:\n{ex}")
            dets = []
        try:
            self.workflow_tab.populate_detection_results(dets)
        except Exception:
            pass
        try:
            self.workflow_tab.append_log(f"[Detectron] {len(dets)} detection(s)")
        except Exception:
            pass
        indexed = self._annotate_top_detections(img_path, dets)
        self._set_top_detections(indexed)
        try:
            idx_list = []
            for d in indexed or []:
                try:
                    idx_list.append(int(d.get("index", 0)))
                except Exception:
                    pass
            self._attachment_count = len(indexed or [])
            try:
                self.defect_ledger.set_total(self._attachment_count)
                if idx_list:
                    self.defect_ledger.set_index_order(idx_list)
            except Exception:
                pass
            try:
                # Seed ledger selection with first attachment
                if idx_list:
                    self.defect_ledger.set_current_index(idx_list[0])
            except Exception:
                pass
        except Exception:
            pass

        # Save annotated overlay for attachments
        try:
            if self._top_raw_np is None and os.path.exists(img_path):
                self._top_raw_np = cv2.imread(img_path)
            if self._top_raw_np is not None:
                annot = self.preview_panel.render_attachment_overlay(np_bgr_to_qpixmap(self._top_raw_np), indexed)
                if annot and not annot.isNull():
                    if ctx.capture_dir is not None:
                        out_path = ctx.capture_dir / "step-01_top_annotated.png"
                        annot.save(str(out_path))
                        self.workflow_tab.append_log(f"[Step1] Saved annotated overlay: {out_path}")
        except Exception:
            pass

        if not indexed:
            QMessageBox.information(
                self.view,
                "Run Detection",
                "No attachments detected in the top view.\n"
                "Consider lowering the detection threshold in model metadata or tuning lighting.",
            )
            return

        # Run steps 2-4
        self._run_step2_sequence(indexed, ctx.capture_dir)

    # ---- Step 2 pipeline: rotate per phi and capture front images ----
    def _run_step2_sequence(self, detections, cap_dir):
        import threading, os, time
        from pathlib import Path
        import math
        from services import turntable_service
        from services import camera_service
        from ui.qt_image import np_bgr_to_qpixmap
        from PyQt5.QtCore import QTimer

        cap_dir = self._normalize_cap_root(cap_dir)
        if cap_dir is None:
            try:
                self.tt_message.emit("[Step2] Capture directory unavailable; skipping rotation sequence.")
            except Exception:
                pass
            return None

        # Prepare folder
        step2_dir = Path(cap_dir) / "step-02"
        try:
            step2_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        step3_dir = Path(cap_dir) / "step-03"
        step4_dir = Path(cap_dir) / "step-04"
        for _d in (step3_dir, step4_dir):
            try:
                _d.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        try:
            self.tt_message.emit("[Step2] Starting rotation and front capture sequence...")
        except Exception:
            pass

        # Order by index
        ordered = []
        try:
            ordered = sorted([d for d in detections if isinstance(d, dict)], key=lambda d: int(d.get("index", 0)))
        except Exception:
            ordered = list(detections)

        def worker():
            try:
                debug_log.log(f"[Step2] worker start cap_dir={cap_dir} count={len(ordered)}")
            except Exception:
                pass
            try:
                cycle_start = float(getattr(self, "_cycle_start_ts", None) or time.time())
            except Exception:
                cycle_start = time.time()
            if not turntable_service.is_connected():
                self.tt_message.emit("[Step2] Turntable not connected; skipping rotation sequence.")
                return
            front_model = solvision_manager.current_project_path_for("front")
            defect_model = solvision_manager.current_project_path_for("defect")
            try:
                from services.config import state as _state

                st_def = _state()
                self._defect_thr_cached = getattr(st_def, "defect_score_threshold", None)
            except Exception:
                self._defect_thr_cached = None
            if not front_model:
                self.tt_message.emit("[Step2] Front model not loaded; skipping rotation/front alignment.")
                return

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

            def _show_top(frame, detections=None):
                try:
                    pm = frame if isinstance(frame, QPixmap) else np_bgr_to_qpixmap(frame)
                    size = None
                    try:
                        size = (pm.width(), pm.height()) if pm is not None and (not pm.isNull()) else None
                    except Exception:
                        size = None
                    from PyQt5.QtCore import QTimer as _QTimer

                    _QTimer.singleShot(0, lambda f=pm, det=detections, sz=size: self.defect_ledger.set_top_pixmap(f, detections=det, image_size=sz))
                except Exception:
                    pass

            last_phi = 0.0
            for d in ordered:
                phi = d.get("phi")
                idx = int(d.get("index", 0) or 0)
                if not isinstance(phi, (int, float)):
                    self.tt_message.emit(f"[Step2] Skipping index {idx}: missing phi.")
                    continue

                move_deg = math.degrees(phi - last_phi)
                last_phi = phi

                target_steps = None
                axis_reason = None
                try:
                    if linear_axis_service.is_calibrated():
                        from services.config import state as _state

                        total_steps = linear_axis_service.calibration_total_steps()
                        if total_steps is None:
                            axis_reason = "invalid calibration (total steps unavailable)"
                            raise RuntimeError(axis_reason)
                        try:
                            off_top = float(d.get("offset_top_rot_px", d.get("offset_top_px", 0.0)) or 0.0)
                        except Exception:
                            off_top = 0.0
                        st_axis = _state()
                        top_fov_val = getattr(st_axis, "front_fov_top_px", None) or DEFAULT_FRONT_FOV_TOP_PX
                        try:
                            top_fov_val = float(top_fov_val)
                        except (TypeError, ValueError):
                            top_fov_val = 0.0
                        if abs(top_fov_val) > 1e-3:
                            delta_px = (off_top / top_fov_val) * FRONT_IMAGE_WIDTH_PX
                            try:
                                cfg = _state()
                                home_steps = getattr(cfg, "linear_axis_home_steps", None)
                                if home_steps is None:
                                    home_steps = int(total_steps) // 2
                            except Exception:
                                home_steps = int(total_steps) // 2
                            delta_steps = int(round(float(delta_px) * float(FRONT_STEPS_PER_PIXEL)))
                            tgt_steps = int(home_steps) + int(delta_steps)
                            target_steps = max(0, min(int(total_steps), int(tgt_steps)))
                        else:
                            axis_reason = "invalid front FOV"
                    else:
                        axis_reason = "linear axis not calibrated"
                except Exception as ex:
                    axis_reason = f"axis alignment failed: {ex}"

                def _axis_move_safe(target: int) -> dict:
                    res = {"msg": None, "err": None}
                    try:
                        curr = linear_axis_service.current_position_steps()
                    except Exception:
                        curr = None
                    if curr is not None and abs(int(curr) - int(target)) <= 2:
                        res["msg"] = "[INFO] Already at requested position."
                        return res
                    try:
                        move_res = linear_axis_service.goto_steps(int(target))
                        if move_res.success:
                            res["msg"] = move_res.message
                        else:
                            low = (move_res.message or "").lower()
                            if ("timed out" in low) or ("timeout" in low):
                                retry = linear_axis_service.goto_steps(int(target))
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
                    except Exception:
                        try:
                            msg = turntable_service.move_relative(delta_deg)
                            res["msg"] = msg + " (retried)"
                        except Exception as ex2:
                            res["err"] = str(ex2)
                    return res

                import threading

                tt_res = {"msg": None, "err": None}
                ax_res = {"msg": None, "err": axis_reason}

                def _move_tt():
                    r = _tt_move_safe(move_deg)
                    tt_res.update(r)

                def _move_axis():
                    if target_steps is None:
                        return
                    r = _axis_move_safe(int(target_steps))
                    ax_res.update(r)

                threads = []
                t1 = threading.Thread(target=_move_tt, daemon=True)
                threads.append(t1)
                t1.start()
                if target_steps is not None:
                    t2 = threading.Thread(target=_move_axis, daemon=True)
                    threads.append(t2)
                    t2.start()

                for t in threads:
                    t.join()

                if tt_res["err"]:
                    self.tt_message.emit(f"[Step2] Rotate idx {idx} failed: {tt_res['err']}")
                    continue
                if tt_res["msg"]:
                    self.tt_message.emit(f"[Step2] Rotate idx {idx}: {tt_res['msg']}")

                if target_steps is None:
                    if axis_reason:
                        self.tt_message.emit(f"[Step2] Axis alignment skipped: {axis_reason}")
                else:
                    if ax_res["err"]:
                        self.tt_message.emit(f"[Step2] Axis alignment failed: {ax_res['err']}")
                    elif ax_res["msg"]:
                        self.tt_message.emit(ax_res["msg"])

                try:
                    time.sleep(0.2)
                except Exception:
                    pass

                try:
                    top_snapshot = None
                    if cammgr.is_connected("Top"):
                        try:
                            top_frame = cammgr.capture("Top")
                            top_snapshot = self._ensure_bgr8(top_frame)
                            _show_top(top_snapshot, detections=self._last_top_detections)
                        except Exception as ex:
                            top_snapshot = None
                            self.tt_message.emit(f"[Step2] Top snapshot failed: {ex}")
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
                            x0 = max(0, cx - half)
                            x1 = min(Wc, cx + half)
                            y0 = max(0, cy - half)
                            y1 = min(Hc, cy + half)
                            crop = img[y0:y1, x0:x1].copy()
                            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                                crop = _cv2.resize(crop, (crop_size, crop_size))
                            return crop

                        overlay = _capture_front()
                        initial_raw_path = None
                        try:
                            initial_raw_path = str(step2_dir / f"step-02_front_initial_{idx:03d}.png")
                            _cv2.imwrite(initial_raw_path, overlay)
                            self.tt_message.emit(f"[Step2] Saved initial front snapshot: {initial_raw_path}")
                        except Exception:
                            initial_raw_path = None

                        try:
                            st2 = _state()
                            crop_size = int(getattr(st2, "step2_crop_size", None) or 1600)
                        except Exception:
                            crop_size = 1600
                        crop = _center_crop(overlay, crop_size)
                        initial_crop_path = str(step2_dir / f"step-02_front_crop_initial_{idx:03d}.png")
                        try:
                            _cv2.imwrite(initial_crop_path, crop)
                        except Exception:
                            pass

                        dets = []
                        try:
                            dets = solvision_manager.detect_for("front", initial_crop_path)
                        except Exception as ex:
                            self.tt_message.emit(f"[Step2] Front detect failed: {ex}")
                            dets = []

                        if not dets:
                            self.tt_message.emit(f"[Step2] No detection in crop idx {idx}; discarding filling.")
                            continue

                        cx_crop = crop.shape[1] / 2.0
                        cy_crop = crop.shape[0] / 2.0

                        def _center_of(det):
                            b = det.get("bounds")
                            if not b:
                                return (None, None)
                            x1, y1, w, h = b
                            return (x1 + w / 2.0, y1 + h / 2.0)

                        det = min(
                            dets,
                            key=lambda dd: float("inf")
                            if _center_of(dd)[0] is None
                            else abs(_center_of(dd)[0] - cx_crop) + abs(_center_of(dd)[1] - cy_crop),
                        )
                        dcx, dcy = _center_of(det)
                        if dcx is None:
                            self.tt_message.emit(f"[Step2] Detection missing center; discarding idx {idx}.")
                            continue
                        dx_px = dcx - cx_crop
                        try:
                            total_steps = linear_axis_service.calibration_total_steps()
                        except Exception:
                            total_steps = None

                        if not total_steps or total_steps <= 0:
                            self.tt_message.emit("[Step2] Correction skipped: actuator calibration invalid (total steps unavailable).")
                        else:
                            try:
                                curr_steps = linear_axis_service.current_position_steps()
                            except Exception:
                                curr_steps = None
                            if curr_steps is None:
                                try:
                                    cfg = _state()
                                    curr_steps = getattr(cfg, "linear_axis_home_steps", None)
                                except Exception:
                                    curr_steps = None
                            if curr_steps is None:
                                curr_steps = int(total_steps) // 2

                            dx_steps = int(round(float(dx_px) * float(FRONT_STEPS_PER_PIXEL)))
                            new_target = max(0, min(int(total_steps), int(curr_steps) - int(dx_steps)))
                            tol_steps = max(1, int(round((0.05 / 100.0) * float(total_steps))))

                            if abs(dx_steps) > tol_steps:
                                try:
                                    corr_res = _axis_move_safe(new_target)
                                    if corr_res["err"]:
                                        self.tt_message.emit(f"[Step2] Correction move failed: {corr_res['err']}")
                                    elif corr_res["msg"]:
                                        self.tt_message.emit(
                                            f"{corr_res['msg']} (correction dx={dx_px:.2f}px -> {dx_steps} steps, new={new_target} steps)"
                                        )
                                except Exception as ex:
                                    self.tt_message.emit(f"[Step2] Correction move failed: {ex}")
                            else:
                                self.tt_message.emit(
                                    f"[Step2] Alignment within tolerance (dx={dx_px:.2f}px -> {dx_steps} steps); no correction move."
                                )

                        try:
                            time.sleep(0.2)
                        except Exception:
                            pass
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

                        try:
                            fn_front = str(step2_dir / f"step-02_front_{idx:03d}.png")
                            if _cv2.imwrite(fn_front, overlay):
                                self.tt_message.emit(f"[Step2] Saved front snapshot (annotated): {fn_front}")
                            else:
                                self.tt_message.emit(f"[Step2] Failed to save front snapshot: {fn_front}")
                        except Exception as ex:
                            self.tt_message.emit(f"[Step2] Save failed: {ex}")

                        try:
                            crops_dir = step2_dir / "step_2_cropped"
                            crops_dir.mkdir(parents=True, exist_ok=True)
                            try:
                                st2 = _state()
                                crop_size = int(getattr(st2, "step2_crop_size", None) or 1600)
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

                        try:
                            from PyQt5.QtCore import QTimer as _QTimer

                            _QTimer.singleShot(0, lambda: self.preview_panel.set_front_markers([]))
                        except Exception:
                            pass

                        if top_snapshot is not None:
                            try:
                                fn_top = str(step2_dir / f"step-02_top_{idx:03d}.png")
                                if _cv2.imwrite(fn_top, top_snapshot):
                                    self.tt_message.emit(f"[Step2] Saved top snapshot: {fn_top}")
                                else:
                                    self.tt_message.emit(f"[Step2] Failed to save top snapshot: {fn_top}")
                            except Exception as ex:
                                self.tt_message.emit(f"[Step2] Top save failed: {ex}")

                        try:
                            pm_front = np_bgr_to_qpixmap(overlay)
                            from PyQt5.QtCore import QTimer as _QTimer

                            _QTimer.singleShot(0, lambda f=pm_front: self.preview_panel.set_front_np(f))
                        except Exception:
                            pass
                    else:
                        self.tt_message.emit("[Step2] Front camera not connected; snapshot skipped.")
                except Exception as ex:
                    self.tt_message.emit(f"[Step2] Snapshot failed: {ex}")

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
                    try:
                        self._run_step3_front(cap_dir)
                    except Exception as ex:
                        self.tt_message.emit(f"[Step3] Failed: {ex}")
                    try:
                        self._run_step4_defect(cap_dir)
                    except Exception as ex:
                        self.tt_message.emit(f"[Step4] Failed: {ex}")
            finally:
                try:
                    exec_bg.shutdown(wait=True)
                except Exception:
                    pass
            try:
                bbox_files = sorted((Path(cap_dir) / "step-03").glob("step-03_front_bbox_*.png"))
                for p in bbox_files:
                    try:
                        import re as _re

                        m = _re.search(r"_(\d+)\.png$", p.name)
                        idx_fallback = int(m.group(1)) if m else 0
                    except Exception:
                        idx_fallback = 0
                    expected = (Path(cap_dir) / "step-04") / f"step-04_defect_{idx_fallback:03d}.png"
                    if expected.exists():
                        continue
                    try:
                        self.tt_message.emit(f"[Step4] Fallback running idx {idx_fallback} from {p.name}")
                    except Exception:
                        pass
                    try:
                        self._process_step4_single(str(p), idx_fallback, Path(cap_dir) / "step-04", defect_model, self._defect_thr_cached)
                    except Exception as ex:
                        try:
                            self.tt_message.emit(f"[Step4] Fallback idx {idx_fallback} failed: {ex}")
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                self._run_step4_defect(cap_dir)
            except Exception as ex:
                try:
                    self.tt_message.emit(f"[Step4] Final sweep failed: {ex}")
                except Exception:
                    pass
            try:
                res = turntable_service.home()
                self.tt_message.emit(res.message)
                status = res.message if res.success else f"Error: {res.message}"
                self.tt_status.emit(status)
            except Exception:
                self.tt_message.emit("[Step2] Home failed.")
            try:
                if linear_axis_service.is_connected() and linear_axis_service.is_calibrated():
                    try:
                        from services.config import state as _state

                        cfg = _state()
                        hs = getattr(cfg, "linear_axis_home_steps", None)
                        if hs is None:
                            total = linear_axis_service.calibration_total_steps()
                            hs = (int(total) // 2) if total else 0
                        home_steps = int(hs)
                    except Exception:
                        total = linear_axis_service.calibration_total_steps()
                        home_steps = (int(total) // 2) if total else 0
                    res_ax_home = linear_axis_service.home(home_steps=home_steps)
                    self.tt_message.emit(res_ax_home.message)
                else:
                    self.tt_message.emit("[Step2] Axis home skipped (not connected/calibrated).")
            except Exception as ex:
                self.tt_message.emit(f"[Step2] Axis home failed: {ex}")
            try:
                elapsed = time.time() - cycle_start
                ct_path = Path(cap_dir) / "cycle_time.txt"
                ct_path.parent.mkdir(parents=True, exist_ok=True)
                with ct_path.open("a", encoding="ascii") as f:
                    f.write(f"{elapsed:.2f}\n")
                self.tt_message.emit(f"[Step2] Cycle time recorded: {elapsed:.2f} s -> {ct_path}")
            except Exception as ex:
                self.tt_message.emit(f"[Step2] Cycle time record failed: {ex}")
            try:
                debug_log.log(f"[Step2] worker done cap_dir={cap_dir}")
            except Exception:
                pass

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
            try:
                Path(step3_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            img = _cv2.imread(str(crop_path))
            if img is None:
                self.tt_message.emit(f"[Step3] idx {idx}: failed to read {crop_path}")
                try:
                    self._update_ledger_front(idx, None, "No attachment detected.")
                except Exception:
                    pass
                return None
            H, W = img.shape[:2]
            dets = []
            try:
                dets = solvision_manager.detect_for("front", str(crop_path))
            except Exception as ex:
                self.tt_message.emit(f"[Step3] idx {idx}: detect failed: {ex}")
                dets = []

            if not dets:
                ann = img.copy()
                _cv2.putText(ann, "No detection", (20, 40), _cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                out_ann = str(step3_dir / f"step-03_front_{idx:03d}.png")
                _cv2.imwrite(out_ann, ann)
                self.tt_message.emit(f"[Step3] idx {idx}: no detection; saved {out_ann}")
                try:
                    self._update_ledger_front(idx, out_ann, "No attachment detected.")
                except Exception:
                    pass
                return None

            cx0, cy0 = W * 0.5, H * 0.5

            def _metric(d):
                try:
                    b = d.get("bounds") or d.get("rect") or None
                    if not b or len(b) < 4:
                        return (float("inf"), -0.0)
                    x, y, w, h = b
                    x = float(x)
                    y = float(y)
                    w = float(w)
                    h = float(h)
                    cx = x + w * 0.5
                    cy = y + h * 0.5
                    dist2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
                    sc = float(d.get("score") or 0.0)
                    return (dist2, -sc)
                except Exception:
                    return (float("inf"), -0.0)

            best = min(dets, key=_metric)
            bx, by, bw, bh = best.get("bounds") or (0, 0, 0, 0)
            try:
                bx = int(round(float(bx)))
                by = int(round(float(by)))
                bw = int(round(float(bw)))
                bh = int(round(float(bh)))
            except Exception:
                bx, by, bw, bh = 0, 0, 0, 0
            bx = max(0, min(W - 1, bx))
            by = max(0, min(H - 1, by))
            bw = max(0, min(W - bx, bw))
            bh = max(0, min(H - by, bh))

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
                        r = int(hs[0:2], 16)
                        g = int(hs[2:4], 16)
                        b = int(hs[4:6], 16)
                        return (b, g, r)
                except Exception:
                    pass
                return (0, 255, 0)

            color = _hex_to_bgr(best.get("color") or "#00FF00")
            _cv2.rectangle(ann, (bx, by), (bx + bw, by + bh), color, 2)
            lbl = f"{best.get('class', '')} {float(best.get('score') or 0.0):.3f}"
            lx, ly = _safe_label_pos(bx, by, bw, bh, lbl)
            _cv2.putText(ann, lbl, (lx, ly), _cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            out_ann = str(step3_dir / f"step-03_front_{idx:03d}.png")
            _cv2.imwrite(out_ann, ann)
            self.tt_message.emit(f"[Step3] idx {idx}: saved {out_ann}")

            pad = 30
            bx2 = max(0, bx - pad)
            by2 = max(0, by - pad)
            bw2 = min(W - bx2, bw + 2 * pad)
            bh2 = min(H - by2, bh + 2 * pad)
            bbox = img[by2 : by2 + bh2, bx2 : bx2 + bw2].copy()
            out_crop = str(step3_dir / f"step-03_front_bbox_{idx:03d}.png")
            _cv2.imwrite(out_crop, bbox)
            return out_crop
        except Exception:
            return None

    def _run_step3_front(self, cap_dir):
        cap_root = self._normalize_cap_root(cap_dir)
        if cap_root is None:
            return None
        step3_dir = Path(cap_root) / "step-03"
        try:
            step3_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        front_path = solvision_manager.current_project_path_for("front")
        crop_dir = Path(cap_root) / "step-02" / "step_2_cropped"
        crop_paths = []
        try:
            for p in sorted(crop_dir.glob("*.png")):
                crop_paths.append(str(p))
        except Exception:
            crop_paths = []

        outputs = []
        for idx, crop_path in enumerate(crop_paths, start=1):
            bbox = self._process_step3_single(crop_path, idx, step3_dir, front_path)
            if bbox:
                outputs.append((idx, bbox))
        return step3_dir

    def _run_step4_defect(self, cap_dir):
        cap_root = self._normalize_cap_root(cap_dir)
        if cap_root is None:
            return None
        step4_dir = Path(cap_root) / "step-04"
        try:
            step4_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return None

        bbox_dir = Path(cap_root) / "step-03"
        bbox_paths = []
        try:
            for p in sorted(bbox_dir.glob("step-03_front_bbox_*.png")):
                bbox_paths.append(str(p))
        except Exception:
            bbox_paths = []

        defect_path = solvision_manager.current_project_path_for("defect")
        thr_override = None
        try:
            thr_override = float(self.workflow_tab.spin_defect_thr.value())
        except Exception:
            thr_override = None

        for idx, bbox_path in enumerate(bbox_paths, start=1):
            self._process_step4_single(bbox_path, idx, step4_dir, defect_path, override_thr=thr_override)
        return step4_dir

    def _process_step4_single(self, bbox_path, idx, step4_dir, defect_path, override_thr=None):
        try:
            if not defect_path:
                self.tt_message.emit("[Step4] No defect model loaded; skipping.")
                return None
            if not os.path.isfile(bbox_path):
                self.tt_message.emit(f"[Step4] idx {idx}: bbox not found: {bbox_path}")
                return None
        except Exception:
            return None

        try:
            try:
                Path(step4_dir).mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

            img = cv2.imread(str(bbox_path))
            if img is None:
                self.tt_message.emit(f"[Step4] idx {idx}: failed to read {bbox_path}")
                return None
            dets = []
            try:
                dets = solvision_manager.detect_for("defect", str(bbox_path), score_threshold=override_thr)
            except Exception as ex:
                self.tt_message.emit(f"[Step4] idx {idx}: detect failed: {ex}")
                dets = []

            ann = img.copy()
            if dets:
                for d in dets:
                    try:
                        x, y, w, h = d.get("bounds") or (0, 0, 0, 0)
                        color = d.get("color") or "#00FF00"
                        (tw, th), _ = cv2.getTextSize(str(d.get("class", "")), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

                        def _hex_to_bgr(hex_str):
                            try:
                                hs = hex_str.lstrip("#")
                                if len(hs) == 6:
                                    r = int(hs[0:2], 16)
                                    g = int(hs[2:4], 16)
                                    b = int(hs[4:6], 16)
                                    return (b, g, r)
                            except Exception:
                                pass
                            return (0, 0, 255)

                        c = _hex_to_bgr(color)
                        cv2.rectangle(ann, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
                        lbl = f"{d.get('class', '')} {float(d.get('score') or 0.0):.3f}"
                        cv2.putText(ann, lbl, (int(x), int(max(0, y - 4))), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
                    except Exception:
                        pass
                self._set_defect_state(idx, "fail")
            else:
                self._set_defect_state(idx, "ok")

            out_ann = str(step4_dir / f"step-04_defect_{idx:03d}.png")
            cv2.imwrite(out_ann, ann)
            self.tt_message.emit(f"[Step4] idx {idx}: saved {out_ann}")
            try:
                self._front_results[int(idx)] = out_ann
            except Exception:
                pass
            try:
                self._update_ledger_front(idx, out_ann)
            except Exception:
                pass
            return out_ann
        except Exception:
            return None

    def run_step3_step4_existing(self):
        path = QFileDialog.getExistingDirectory(self.view, "Select existing capture run", str(app_root()))
        if not path:
            return
        cap_root = self._normalize_cap_root(path)
        if cap_root is None:
            QMessageBox.warning(self.view, "Run Step 3/4", "Invalid capture folder selected.")
            return
        step2_dir = Path(cap_root) / "step-02"
        if not step2_dir.exists():
            QMessageBox.warning(self.view, "Run Step 3/4", "Selected folder does not contain step-02 outputs.")
            return
        if not self.models.ensure_models_loaded(
            required=("front", "defect"),
            parent=self.view,
            show_dialog=True,
            log_callback=self.workflow_tab.append_log,
        ):
            return
        self._run_step3_front(cap_root)
        self._run_step4_defect(cap_root)
        try:
            self._last_cap_root = cap_root
            self._cycle_cap_dir = cap_root
        except Exception:
            pass
