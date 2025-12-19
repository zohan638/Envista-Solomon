from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
)

from .camera_panel import CameraPanel
from .image_preview_panel import ImagePreviewPanel
from .plc_panel import PlcPanel
from services import camera_service, plc_service
from services.config import settings, state, save_state


class InitWizard(QDialog):
    live_frame_ready = pyqtSignal(str, int, object)  # (role, gen, frame)
    live_error_ready = pyqtSignal(str, int, str)  # (role, gen, err_short)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Initialization")
        self.setModal(True)
        self.resize(780, 540)

        root = QVBoxLayout(self)

        # Files group
        files = QGroupBox("Step 1 - Select Models/Files")
        vfiles = QVBoxLayout(files)

        def row(label: str):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            le = QLineEdit()
            btn = QPushButton("Browse...")
            h.addWidget(le, stretch=1)
            h.addWidget(btn)
            return h, le, btn

        r1, self.le_attach, b1 = row("Attachment:")
        r2, self.le_front, b2 = row("Front Attachment:")
        r3, self.le_defect, b3 = row("Defect:")
        vfiles.addLayout(r1)
        vfiles.addLayout(r2)
        vfiles.addLayout(r3)

        def pick(le: QLineEdit):
            path, _ = QFileDialog.getOpenFileName(self, "Select file", "", "All Files (*.*)")
            if path:
                le.setText(path)

        b1.clicked.connect(lambda: pick(self.le_attach))
        b2.clicked.connect(lambda: pick(self.le_front))
        b3.clicked.connect(lambda: pick(self.le_defect))

        root.addWidget(files)

        # Cameras group
        self.cam_panel = CameraPanel()
        root.addWidget(self.cam_panel)

        # Live preview
        self.preview_panel = ImagePreviewPanel()
        root.addWidget(self.preview_panel)

        # PLC connection
        self.plc_panel = PlcPanel()
        root.addWidget(self.plc_panel)

        # Bottom row
        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self.bt_begin = QPushButton("Begin Workflow")
        self.bt_begin.setEnabled(False)
        self.bt_cancel = QPushButton("Cancel")
        bottom.addWidget(self.bt_cancel)
        bottom.addWidget(self.bt_begin)
        root.addLayout(bottom)

        # Live camera feed (no manual capture button)
        import concurrent.futures

        self._live_enabled = False
        self._live_closed = False
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(50)
        self._live_timer.timeout.connect(self._on_live_tick)
        self._live_gen = {"Top": 0, "Front": 0}
        self._live_inflight = {"Top": None, "Front": None}
        self._live_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.live_frame_ready.connect(self._on_live_frame_ready)
        self.live_error_ready.connect(self._on_live_error_ready)

        # Wire cameras
        self.cam_panel.refresh_requested.connect(self.on_cam_refresh)
        self.cam_panel.connect_requested.connect(self.on_cam_connect)
        self.cam_panel.disconnect_requested.connect(self.on_cam_disconnect)
        self.on_cam_refresh()

        # Wire PLC
        self.plc_panel.refresh_requested.connect(self.on_plc_refresh)
        self.plc_panel.connect_requested.connect(self.on_plc_connect)
        self.on_plc_refresh()

        # Buttons
        self.bt_cancel.clicked.connect(self.reject)
        self.bt_begin.clicked.connect(self.accept)

        # Pre-fill from persisted state
        st = state()
        if st.attachment_path:
            self.le_attach.setText(st.attachment_path)
        if st.front_attachment_path:
            self.le_front.setText(st.front_attachment_path)
        if st.defect_path:
            self.le_defect.setText(st.defect_path)

        # Validation timerless check
        self._update_ready()
        self.le_attach.textChanged.connect(self._update_ready)
        self.le_front.textChanged.connect(self._update_ready)
        self.le_defect.textChanged.connect(self._update_ready)

    def _on_live_frame_ready(self, role: str, gen: int, frame):
        if self._live_closed or not self._live_enabled:
            return
        role_norm = "Top" if role == "Top" else "Front"
        try:
            if int(self._live_gen.get(role_norm, 0) or 0) != int(gen):
                return
        except Exception:
            return
        if frame is None:
            return
        try:
            from .qt_image import np_bgr_to_qpixmap

            pm = np_bgr_to_qpixmap(frame)
            if pm is None or pm.isNull():
                return
            if role_norm == "Top":
                self.preview_panel.set_original_np(pm)
                settings().top_preview_np = frame
            else:
                self.preview_panel.set_front_np(pm)
                settings().front_preview_np = frame
            try:
                self.cam_panel.set_stream_status(role_norm, "Live feed: OK")
            except Exception:
                pass
        except Exception:
            return

    def _on_live_error_ready(self, role: str, gen: int, err_short: str):
        if self._live_closed or not self._live_enabled:
            return
        role_norm = "Top" if role == "Top" else "Front"
        try:
            if int(self._live_gen.get(role_norm, 0) or 0) != int(gen):
                return
        except Exception:
            return
        try:
            self.cam_panel.set_stream_status(role_norm, f"Live feed: error ({err_short})")
        except Exception:
            pass

    # Cameras
    def on_cam_refresh(self):
        try:
            devices = camera_service.enumerate_devices()
            self.cam_panel.set_devices(devices)
            # Restore last selections (no auto-connect)
            st = state()
            if st.camera_top_index is not None:
                self.cam_panel.set_selected_index("Top", int(st.camera_top_index))
            if st.camera_front_index is not None:
                self.cam_panel.set_selected_index("Front", int(st.camera_front_index))
        except Exception:
            pass

    def on_cam_connect(self, role: str, index: int):
        # prevent same device for both roles
        other = "Front" if role == "Top" else "Top"
        if camera_service.get_connected_index(other) == index:
            return
        if camera_service.connect(role, index):
            name = ""
            self.cam_panel.set_connected(role, True, name)
            # Persist selection
            st = state()
            if role == "Top":
                st.camera_top_index = index
            else:
                st.camera_front_index = index
            save_state()
            self._start_live()
        self._update_ready()

    def on_cam_disconnect(self, role: str):
        camera_service.disconnect(role)
        self.cam_panel.set_connected(role, False)
        self._bump_live(role)
        self._stop_live_if_idle()
        self._update_ready()

    def _bump_live(self, role: str):
        try:
            self._live_gen["Top" if role == "Top" else "Front"] += 1
        except Exception:
            pass

    def _start_live(self):
        if self._live_closed:
            return
        self._live_enabled = True
        try:
            if not self._live_timer.isActive():
                self._live_timer.start()
        except Exception:
            pass

    def _stop_live_if_idle(self):
        try:
            if not camera_service.is_connected("Top") and not camera_service.is_connected("Front"):
                self._live_enabled = False
                self._live_timer.stop()
        except Exception:
            pass

    def _on_live_tick(self):
        if self._live_closed or not self._live_enabled:
            return

        top_ok = bool(camera_service.is_connected("Top"))
        front_ok = bool(camera_service.is_connected("Front"))
        if not top_ok and not front_ok:
            self._stop_live_if_idle()
            return

        def _schedule(role: str):
            if not camera_service.is_connected(role):
                return
            fut = self._live_inflight.get(role)
            if fut is not None and not fut.done():
                return
            gen = int(self._live_gen.get(role, 0) or 0)
            from services import camera_manager as _cammgr

            fut = self._live_executor.submit(_cammgr.capture_live, role)
            self._live_inflight[role] = fut

            def _done(_fut, role_inner=role, gen_inner=gen):
                try:
                    self._live_inflight[role_inner] = None
                except Exception:
                    pass
                if self._live_closed:
                    return
                try:
                    frame = _fut.result()
                except Exception:
                    try:
                        err = str(_fut.exception() or "capture failed")
                        err_short = str(err).splitlines()[-1].strip()
                    except Exception:
                        err_short = "capture failed"
                    try:
                        self.live_error_ready.emit(role_inner, int(gen_inner), str(err_short))
                    except Exception:
                        pass
                    return
                if frame is None:
                    return
                try:
                    self.live_frame_ready.emit(role_inner, int(gen_inner), frame)
                except Exception:
                    pass

            try:
                fut.add_done_callback(_done)
            except Exception:
                pass

        _schedule("Top")
        _schedule("Front")

    def _shutdown_live(self):
        self._live_closed = True
        self._live_enabled = False
        try:
            self._live_timer.stop()
        except Exception:
            pass
        try:
            self._live_executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                self._live_executor.shutdown(wait=False)
            except Exception:
                pass

    # PLC
    def on_plc_refresh(self):
        try:
            # Reuse the motion service host list as the source of known endpoints.
            from services import turntable_service

            hosts = turntable_service.refresh_devices()
            self.plc_panel.set_hosts(hosts)
            st = state()
            if getattr(st, "plc_host", None):
                idx = self.plc_panel.host_combo.findText(str(st.plc_host))
                if idx >= 0:
                    self.plc_panel.host_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def on_plc_connect(self, host: str):
        # Allow "host:port" input for convenience.
        host_str = str(host or "").strip()
        port_override = None
        if ":" in host_str and host_str.count(":") == 1:
            h, p = host_str.split(":", 1)
            host_str = h.strip()
            try:
                port_override = int(p.strip())
            except Exception:
                port_override = None

        st = state()
        plc_port = int(getattr(st, "plc_port", 502) or 502)
        unit_id = int(getattr(st, "plc_unit_id", 255) or 255)
        if port_override is not None:
            plc_port = int(port_override)

        if plc_service.connect(host_str, port=plc_port, unit_id=unit_id, force_reconnect=True):
            try:
                st.plc_host = host_str
                st.plc_port = plc_port
                st.plc_unit_id = unit_id
                save_state()
            except Exception:
                pass
            self.plc_panel.set_connected(True, plc_service.endpoint() or host_str)
        else:
            self.plc_panel.set_connected(False, plc_service.last_error() or "PLC connection failed.")
        self._update_ready()

    def _update_ready(self):
        # Persist into settings
        s = settings()
        s.attachment_path = self.le_attach.text().strip() or None
        s.front_attachment_path = self.le_front.text().strip() or None
        s.defect_path = self.le_defect.text().strip() or None
        st = state()
        st.attachment_path = s.attachment_path
        st.front_attachment_path = s.front_attachment_path
        st.defect_path = s.defect_path
        save_state()

        cams_ok = (
            camera_service.get_connected_index("Top") is not None and
            camera_service.get_connected_index("Front") is not None
        )
        plc_ok = plc_service.is_connected()
        # Files optional for now
        self.bt_begin.setEnabled(cams_ok and plc_ok)

    def accept(self):
        # Save final selections before closing
        try:
            self._shutdown_live()
        except Exception:
            pass
        st = state()
        st.attachment_path = self.le_attach.text().strip() or None
        st.front_attachment_path = self.le_front.text().strip() or None
        st.defect_path = self.le_defect.text().strip() or None
        save_state()
        super().accept()

    def reject(self):
        try:
            self._shutdown_live()
        except Exception:
            pass
        super().reject()
