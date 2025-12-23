import os

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QWidget, QSplitter, QVBoxLayout, QTabWidget, QMessageBox

from controllers.hardware_controller import HardwareController
from controllers.live_camera_controller import LiveCameraController
from controllers.model_controller import ModelController
from controllers.workflow_controller import WorkflowController
from services import camera_service, plc_service, turntable_service, linear_axis_service
from services.config import save_state, settings, state
from ui.defect_ledger import DefectLedger
from ui.image_preview_panel import ImagePreviewPanel
from ui.logic_tab import LogicTab
from ui.workflow_tab import WorkflowTab
from ui.qt_image import np_bgr_to_qpixmap


class _AxisUiBridge(QWidget):
    """Thread-safe hooks to update the linear axis panel from controller signals."""

    def __init__(self, panel, parent=None):
        super().__init__(parent)
        self.panel = panel

    def set_ready(self, ready: bool):
        try:
            self.panel.set_ready(ready)
        except Exception:
            pass

    def set_calibrating(self, calibrating: bool):
        try:
            self.panel.set_calibrating(calibrating)
        except Exception:
            pass

    def set_calibrated(self, ok: bool, position_steps=None, total_steps=None):
        try:
            self.panel.set_calibrated(ok, position_steps, total_steps=total_steps)
        except Exception:
            pass

    def set_position(self, pos):
        try:
            self.panel.set_position(pos)
        except Exception:
            pass


class MainWindow(QMainWindow):
    tt_message = pyqtSignal(str)
    tt_status = pyqtSignal(str)
    plc_snapshot = pyqtSignal(object)

    def __init__(
        self,
        live_controller: LiveCameraController,
        hardware_controller: HardwareController,
        model_controller: ModelController,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Detectron Demo")
        self.resize(1184, 661)

        self.live = live_controller
        self.hardware = hardware_controller
        self.models = model_controller

        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)

        self.tabs = QTabWidget()
        self.workflow_tab = WorkflowTab()
        self.logic_tab = LogicTab()
        self.tabs.addTab(self.workflow_tab, "Workflow")
        self.tabs.addTab(self.logic_tab, "Logic")
        root_splitter.addWidget(self.tabs)

        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.preview_panel = ImagePreviewPanel()
        right_layout.addWidget(self.preview_panel)

        self.defect_ledger = DefectLedger()
        right_layout.addWidget(self.defect_ledger)

        root_splitter.addWidget(right_container)
        root_splitter.setStretchFactor(0, 38)
        root_splitter.setStretchFactor(1, 62)
        self.setCentralWidget(root_splitter)

        # Controllers (after views exist)
        self.workflow_controller = WorkflowController(self, self.live, self.hardware, self.models)

        try:
            self.defect_ledger.prev_requested.connect(self.workflow_controller.on_prev)
            self.defect_ledger.next_requested.connect(self.workflow_controller.on_next)
            self.defect_ledger.selection_changed.connect(self.workflow_controller.on_ledger_selection_changed)
        except Exception:
            pass

        # Wire workflow actions to controller
        self.workflow_tab.load_image_requested.connect(self.workflow_controller.load_image)
        self.workflow_tab.run_detection_requested.connect(self.workflow_controller.run_detection)
        self.workflow_tab.open_tuner_requested.connect(self.workflow_controller.open_tuner)
        self.workflow_tab.load_attachment_requested.connect(self.workflow_controller.load_attachment_file)
        self.workflow_tab.load_front_requested.connect(self.workflow_controller.load_front_file)
        self.workflow_tab.load_defect_requested.connect(self.workflow_controller.load_defect_file)
        self.workflow_tab.run_step3_step4_requested.connect(self.workflow_controller.run_step3_step4_existing)
        try:
            self.workflow_tab.defect_threshold_changed.connect(self.workflow_controller.on_defect_threshold_changed)
        except Exception:
            pass
        try:
            self.preview_panel.attachment_clicked.connect(self.workflow_controller.on_attachment_clicked)
        except Exception:
            pass

        # Camera panel
        cam = self.workflow_tab.camera_panel
        cam.refresh_requested.connect(self.on_camera_refresh)
        cam.connect_requested.connect(self.on_camera_connect)
        cam.disconnect_requested.connect(self.on_camera_disconnect)
        cam.selection_changed.connect(self.on_camera_selected)

        # Turntable panel
        tt = self.workflow_tab.turntable_panel
        tt.refresh_requested.connect(self.on_turntable_refresh)
        tt.connect_requested.connect(self.on_turntable_connect)
        tt.home_requested.connect(self.on_turntable_home)
        tt.rotate_requested.connect(self.on_turntable_rotate)
        tt.port_selected.connect(self.on_turntable_port_selected)
        tt.step_changed.connect(self.on_turntable_step_changed)

        # Linear axis panel
        ax = self.workflow_tab.linear_axis_panel
        ax.refresh_requested.connect(self.on_axis_refresh)
        ax.connect_requested.connect(self.on_axis_connect)
        ax.calibrate_requested.connect(self.on_axis_calibrate)
        ax.home_requested.connect(self.on_axis_home)
        ax.goto_requested.connect(self.on_axis_goto)
        ax.home_set_requested.connect(self.on_axis_home_set)
        self._axis_bridge = _AxisUiBridge(ax, self)
        self.hardware.axis_ready.connect(self._axis_bridge.set_ready)
        self.hardware.axis_calibrating.connect(self._axis_bridge.set_calibrating)
        self.hardware.axis_calibrated.connect(lambda ok, p=None, t=None: self._axis_bridge.set_calibrated(ok, p, t))
        self.hardware.axis_position.connect(self._axis_bridge.set_position)
        self.hardware.axis_log.connect(self.workflow_tab.append_log)

        # Hardware signals
        self.hardware.turntable_message.connect(self._handle_turntable_message)
        self.hardware.turntable_status.connect(self._handle_turntable_status)
        self.hardware.plc_snapshot.connect(self._handle_plc_snapshot)

        # Live camera signals
        self.live.frame_ready.connect(self._on_live_frame_ready)
        self.live.error_ready.connect(self._on_live_error_ready)
        self.live.devices_refreshed.connect(self._on_devices_refreshed)

        # Seed camera list and state
        self.on_camera_refresh()
        self._restore_camera_state()
        self._restore_axis_state()
        self._restore_turntable_state()

        # Seed previews from wizard if available
        try:
            s = settings()
            st = state()
            attach_path = st.attachment_path or getattr(st, "last_project_path", None)
            front_path = st.front_attachment_path or getattr(st, "last_front_project_path", None)
            defect_path = st.defect_path or getattr(st, "last_defect_project_path", None)
            self.workflow_tab.set_selected_files(attach_path, front_path, defect_path)
            if s.top_preview_np is not None:
                pm = np_bgr_to_qpixmap(s.top_preview_np)
                self.preview_panel.set_original_np(pm)
            if s.front_preview_np is not None:
                pm = np_bgr_to_qpixmap(s.front_preview_np)
                self.preview_panel.set_front_np(pm)
        except Exception:
            pass

        # Auto-load any previously saved model paths (best-effort, async).
        # Heavy model auto-load on startup can freeze the UI; gate behind env flag.
        try:
            if os.environ.get("ENVISTA_AUTOLOAD_MODELS", "0") == "1":
                self._auto_load_saved_models()
        except Exception:
            pass

        # Start live feed if anything connected
        try:
            if camera_service.is_connected("Top") or camera_service.is_connected("Front"):
                self.live.start_live()
        except Exception:
            pass

    # --- Live feed slots ----------------------------------------------------
    def _on_live_frame_ready(self, role: str, gen: int, frame):
        if frame is None:
            return
        try:
            pm = np_bgr_to_qpixmap(frame)
            if pm is None or pm.isNull():
                return
            role_norm = "Top" if role == "Top" else "Front"
            if role_norm == "Top":
                self.preview_panel.set_original_np(pm)
            else:
                self.preview_panel.set_front_np(pm)
            self.workflow_tab.camera_panel.set_stream_status(role_norm, "Live feed: OK")
        except Exception:
            pass

    def _on_live_error_ready(self, role: str, gen: int, err_short: str, err_full: str):
        try:
            role_norm = "Top" if role == "Top" else "Front"
            self.workflow_tab.append_log(f"[Camera] Live {role_norm} failed: {err_full}")
            self.workflow_tab.camera_panel.set_stream_status(role_norm, f"Live feed: error ({err_short})")
        except Exception:
            pass

    def _on_devices_refreshed(self, devices):
        try:
            self.workflow_tab.camera_panel.set_devices(devices)
        except Exception:
            pass

    # --- Camera actions -----------------------------------------------------
    def on_camera_refresh(self):
        devices = self.live.refresh_devices()
        try:
            self.workflow_tab.camera_panel.set_devices(devices)
        except Exception:
            pass
        st = state()
        try:
            if st.camera_top_index is not None:
                self.workflow_tab.camera_panel.set_selected_index("Top", int(st.camera_top_index))
            if st.camera_front_index is not None:
                self.workflow_tab.camera_panel.set_selected_index("Front", int(st.camera_front_index))
        except Exception:
            pass

    def on_camera_connect(self, role: str, index: int):
        ok, msg = self.live.connect_camera(role, index)
        self.workflow_tab.camera_panel.set_connected(role, ok, msg if ok else "")
        if not ok and msg:
            QMessageBox.information(self, "Camera", msg)
        else:
            self.live.start_live()

    def on_camera_disconnect(self, role: str):
        self.live.disconnect_camera(role)
        self.workflow_tab.camera_panel.set_connected(role, False)
        self.live.stop_if_idle()

    def on_camera_selected(self, role: str, index: int):
        try:
            st = state()
            if role == "Top":
                st.camera_top_index = index
            else:
                st.camera_front_index = index
            save_state()
        except Exception:
            pass

    # --- Turntable actions --------------------------------------------------
    def on_turntable_refresh(self):
        try:
            hosts = self.hardware.refresh_turntable_devices()
            self.workflow_tab.turntable_panel.set_ports(hosts)
            st = state()
            if getattr(st, "plc_host", None):
                idx = self.workflow_tab.turntable_panel.port_combo.findText(str(st.plc_host))
                if idx >= 0:
                    self.workflow_tab.turntable_panel.port_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def on_turntable_connect(self, port: str):
        ok, msg = self.hardware.connect_turntable(port)
        self.workflow_tab.turntable_panel.set_connected(ok, turntable_service.port_name() or port)
        if not ok:
            QMessageBox.warning(self, "Turntable", msg)

    def on_turntable_home(self):
        def run():
            res_ok, res_msg, _ = self.hardware.home_turntable()
            try:
                self.tt_message.emit(res_msg)
                self.tt_status.emit(res_msg if res_ok else f"Error: {res_msg}")
            except Exception:
                pass

        import threading

        threading.Thread(target=run, daemon=True).start()

    def on_turntable_rotate(self, angle: float):
        import threading

        def run():
            ok, msg = self.hardware.rotate_turntable(angle)
            try:
                self.tt_message.emit(msg)
            except Exception:
                pass
            if not ok:
                try:
                    self.tt_status.emit(msg)
                except Exception:
                    pass

        threading.Thread(target=run, daemon=True).start()

    def on_turntable_port_selected(self, port: str):
        try:
            host = str(port or "").strip()
            if ":" in host and host.count(":") == 1:
                host = host.split(":", 1)[0].strip()
            st = state()
            st.plc_host = host or None
            st.turntable_step = float(self.workflow_tab.turntable_panel.step.value())
            save_state()
        except Exception:
            pass

    def on_turntable_step_changed(self, v: float):
        try:
            st = state()
            st.turntable_step = float(v)
            save_state()
        except Exception:
            pass

    # --- Axis actions -------------------------------------------------------
    def on_axis_refresh(self):
        try:
            ports = self.hardware.refresh_axis_devices()
            self.workflow_tab.linear_axis_panel.set_ports(ports)
            st = state()
            if getattr(st, "plc_host", None):
                combo = self.workflow_tab.linear_axis_panel.port_combo
                idx = combo.findText(str(st.plc_host))
                if idx >= 0:
                    combo.setCurrentIndex(idx)
        except Exception as ex:
            self.workflow_tab.append_log(f"[Axis] Refresh failed: {ex}")

    def on_axis_connect(self, port: str):
        ok, endpoint, home_steps = self.hardware.connect_axis(port)
        self.workflow_tab.linear_axis_panel.set_connected(ok, endpoint)
        self.workflow_tab.linear_axis_panel.set_ready(ok)
        if ok:
            self.workflow_tab.append_log(f"[PLC] Connected to {endpoint}.")
            try:
                self.workflow_tab.linear_axis_panel.set_home_steps(int(home_steps) if home_steps is not None else 0)
            except Exception:
                pass
            try:
                cal = linear_axis_service.is_calibrated()
                pos_steps = linear_axis_service.current_position_steps()
                total_steps = linear_axis_service.calibration_total_steps()
                self.workflow_tab.linear_axis_panel.set_calibrated(bool(cal), pos_steps if cal else None, total_steps=total_steps)
            except Exception:
                pass
        else:
            self.workflow_tab.append_log(f"[PLC] Connection failed for {port}")

    def on_axis_calibrate(self):
        try:
            home_steps = int(self.workflow_tab.linear_axis_panel.home_steps())
        except Exception:
            home_steps = 0
        self.hardware.calibrate_axis(home_steps)

    def on_axis_home(self, home_steps: int):
        self.hardware.home_axis(int(home_steps))

    def on_axis_goto(self, target_steps: int):
        self.hardware.goto_axis(int(target_steps))

    def on_axis_home_set(self, home_steps: int):
        self.hardware.set_axis_home(int(home_steps))

    # --- PLC snapshot handling ---------------------------------------------
    def _handle_plc_snapshot(self, snap):
        try:
            connected = bool(getattr(snap, "connected", False))
            endpoint = plc_service.endpoint() or ""
            self.workflow_tab.turntable_panel.set_connected(connected, endpoint)
            self.workflow_tab.linear_axis_panel.set_connected(connected, endpoint)
            self.workflow_tab.linear_axis_panel.set_ready(connected)
            if not connected:
                err = getattr(snap, "last_error", None) or "PLC disconnected."
                self.workflow_tab.turntable_panel.set_status(str(err))
                self.workflow_tab.linear_axis_panel.set_status(str(err))
                return
            status = getattr(snap, "status", None)
            if status:
                try:
                    act_pos = getattr(status, "act_pos_steps", None)
                    act_target = getattr(status, "act_target_steps", None)
                    act_in_motion = getattr(status, "act_in_motion", None)
                    act_state = getattr(status, "act_state", None)
                    act_fault = getattr(status, "act_fault_code", None)
                    act_cal = getattr(status, "act_calib_valid", None)
                    act_total = getattr(status, "act_calib_total_steps", None)
                    self.workflow_tab.linear_axis_panel.set_plc_axis_snapshot(
                        position_steps=act_pos,
                        target_steps=act_target,
                        in_motion=act_in_motion,
                        act_state=act_state,
                        act_fault_code=act_fault,
                        calibrated=act_cal,
                        total_steps=act_total,
                    )
                    if act_cal:
                        self.workflow_tab.linear_axis_panel.set_calibrated(True, position_steps=act_pos, total_steps=act_total)
                    ready = bool(getattr(status, "sys_ready", False))
                    self.workflow_tab.linear_axis_panel.set_ready(ready)
                except Exception:
                    pass
        except Exception:
            pass

    # --- Turntable message relays ------------------------------------------
    def _handle_turntable_message(self, msg: str):
        try:
            if msg:
                self.workflow_tab.append_log(msg)
        except Exception:
            pass

    def _handle_turntable_status(self, status: str):
        try:
            if status:
                self.workflow_tab.turntable_panel.set_status(status)
        except Exception:
            pass

    # --- Restore persisted UI state ----------------------------------------
    def _restore_camera_state(self):
        try:
            top_idx = camera_service.get_connected_index("Top")
            front_idx = camera_service.get_connected_index("Front")
            if top_idx is not None:
                self.workflow_tab.camera_panel.set_connected("Top", True, "")
            if front_idx is not None:
                self.workflow_tab.camera_panel.set_connected("Front", True, "")
            st = state()
            if top_idx is None and st.camera_top_index is not None:
                self.workflow_tab.camera_panel.set_selected_index("Top", int(st.camera_top_index))
            if front_idx is None and st.camera_front_index is not None:
                self.workflow_tab.camera_panel.set_selected_index("Front", int(st.camera_front_index))
        except Exception:
            pass

    def _restore_turntable_state(self):
        try:
            if turntable_service.is_connected():
                self.workflow_tab.turntable_panel.set_connected(True, turntable_service.port_name())
            else:
                st = state()
                if getattr(st, "plc_host", None):
                    idx = self.workflow_tab.turntable_panel.port_combo.findText(str(st.plc_host))
                    if idx >= 0:
                        self.workflow_tab.turntable_panel.port_combo.setCurrentIndex(idx)
                if st.turntable_step is not None:
                    self.workflow_tab.turntable_panel.step.setValue(float(st.turntable_step))
        except Exception:
            pass

    def _restore_axis_state(self):
        try:
            st = state()
            hs = getattr(st, "linear_axis_home_steps", None)
            if hs is None:
                hs = getattr(st, "linear_axis_last_steps", None)
            if hs is None:
                hs = 0
            self.workflow_tab.linear_axis_panel.set_home_steps(int(hs))
        except Exception:
            pass

    def _auto_load_saved_models(self):
        st = state()
        attach = getattr(st, "attachment_path", None) or getattr(st, "last_project_path", None)
        front = getattr(st, "front_attachment_path", None) or getattr(st, "last_front_project_path", None)
        defect = getattr(st, "defect_path", None) or getattr(st, "last_defect_project_path", None)

        if attach:
            self.workflow_controller.load_attachment_file(attach)
        if front:
            self.workflow_controller.load_front_file(front)
        if defect:
            self.workflow_controller.load_defect_file(defect)

        # Fallback: if still missing, reflect any already-loaded solvision paths into UI and state
        try:
            from services import solvision_manager

            loaded_top = solvision_manager.current_project_path_for("top")
            loaded_front = solvision_manager.current_project_path_for("front")
            loaded_defect = solvision_manager.current_project_path_for("defect")
            updated = False
            if not attach and loaded_top:
                st.attachment_path = loaded_top
                updated = True
            if not front and loaded_front:
                st.front_attachment_path = loaded_front
                st.last_front_project_path = loaded_front
                updated = True
            if not defect and loaded_defect:
                st.defect_path = loaded_defect
                st.last_defect_project_path = loaded_defect
                updated = True
            if updated:
                save_state()
                self.workflow_tab.set_selected_files(
                    st.attachment_path or getattr(st, "last_project_path", None),
                    st.front_attachment_path or getattr(st, "last_front_project_path", None),
                    st.defect_path or getattr(st, "last_defect_project_path", None),
                )
        except Exception:
            pass

    # --- Close --------------------------------------------------------------
    def closeEvent(self, event):
        try:
            self.live.shutdown()
        except Exception:
            pass
        try:
            camera_service.release_all()
        except Exception:
            pass
        try:
            self.hardware.shutdown()
        except Exception:
            pass
        try:
            plc_service.disconnect()
        except Exception:
            pass
        super().closeEvent(event)
