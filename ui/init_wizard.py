from PyQt5.QtCore import Qt
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

from controllers.hardware_controller import HardwareController
from controllers.live_camera_controller import LiveCameraController
from controllers.model_controller import ModelController
from services import camera_service, plc_service
from services.config import settings, state, save_state
from ui.camera_panel import CameraPanel
from ui.image_preview_panel import ImagePreviewPanel
from ui.plc_panel import PlcPanel
from ui.qt_image import np_bgr_to_qpixmap


class InitWizard(QDialog):
    def __init__(
        self,
        live_controller: LiveCameraController,
        hardware_controller: HardwareController,
        model_controller: ModelController,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Initialization")
        self.setModal(True)
        self.resize(780, 540)

        self.live = live_controller
        self.hardware = hardware_controller
        self.models = model_controller

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
        if st.attachment_path or st.last_project_path:
            self.le_attach.setText(st.attachment_path or st.last_project_path)
        if st.front_attachment_path or getattr(st, "last_front_project_path", None):
            self.le_front.setText(st.front_attachment_path or getattr(st, "last_front_project_path", None))
        if st.defect_path or getattr(st, "last_defect_project_path", None):
            self.le_defect.setText(st.defect_path or getattr(st, "last_defect_project_path", None))

        self.le_attach.textChanged.connect(self._update_ready)
        self.le_front.textChanged.connect(self._update_ready)
        self.le_defect.textChanged.connect(self._update_ready)

        # Live signals
        self.live.frame_ready.connect(self._on_live_frame_ready)
        self.live.error_ready.connect(self._on_live_error_ready)
        try:
            if camera_service.is_connected("Top") or camera_service.is_connected("Front"):
                self.live.start_live()
        except Exception:
            pass

    # Live handling
    def _on_live_frame_ready(self, role: str, gen: int, frame):
        if frame is None:
            return
        role_norm = "Top" if role == "Top" else "Front"
        try:
            pm = np_bgr_to_qpixmap(frame)
            if pm is None or pm.isNull():
                return
            if role_norm == "Top":
                self.preview_panel.set_original_np(pm)
                settings().top_preview_np = frame
            else:
                self.preview_panel.set_front_np(pm)
                settings().front_preview_np = frame
            self.cam_panel.set_stream_status(role_norm, "Live feed: OK")
        except Exception:
            pass

    def _on_live_error_ready(self, role: str, gen: int, err_short: str, err_full: str):
        try:
            role_norm = "Top" if role == "Top" else "Front"
            self.cam_panel.set_stream_status(role_norm, f"Live feed: error ({err_short})")
        except Exception:
            pass

    # Cameras
    def on_cam_refresh(self):
        devices = self.live.refresh_devices()
        self.cam_panel.set_devices(devices)
        st = state()
        if st.camera_top_index is not None:
            self.cam_panel.set_selected_index("Top", int(st.camera_top_index))
        if st.camera_front_index is not None:
            self.cam_panel.set_selected_index("Front", int(st.camera_front_index))
        self._update_ready()

    def on_cam_connect(self, role: str, index: int):
        ok, msg = self.live.connect_camera(role, index)
        if ok:
            self.cam_panel.set_connected(role, True, "")
            self.live.start_live()
        else:
            self.cam_panel.set_connected(role, False)
        self._update_ready()

    def on_cam_disconnect(self, role: str):
        self.live.disconnect_camera(role)
        self.cam_panel.set_connected(role, False)
        self.live.stop_if_idle()
        self._update_ready()

    # PLC
    def on_plc_refresh(self):
        try:
            hosts = self.hardware.refresh_turntable_devices()
            self.plc_panel.set_hosts(hosts)
            st = state()
            if getattr(st, "plc_host", None):
                idx = self.plc_panel.host_combo.findText(str(st.plc_host))
                if idx >= 0:
                    self.plc_panel.host_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def on_plc_connect(self, host: str):
        ok, msg = self.hardware.connect_turntable(host)
        endpoint = plc_service.endpoint() or host
        self.plc_panel.set_connected(ok, endpoint if ok else (msg or "PLC connection failed."))
        self._update_ready()

    def _update_ready(self):
        s = settings()
        st = state()

        attach_text = self.le_attach.text().strip()
        front_text = self.le_front.text().strip()
        defect_text = self.le_defect.text().strip()

        if attach_text:
            s.attachment_path = attach_text
            st.attachment_path = attach_text
            st.last_project_path = attach_text
        if front_text:
            s.front_attachment_path = front_text
            st.front_attachment_path = front_text
            st.last_front_project_path = front_text
        if defect_text:
            s.defect_path = defect_text
            st.defect_path = defect_text
            st.last_defect_project_path = defect_text
        save_state()

        cams_ok = camera_service.get_connected_index("Top") is not None and camera_service.get_connected_index("Front") is not None
        plc_ok = plc_service.is_connected()
        self.bt_begin.setEnabled(cams_ok and plc_ok)

    def accept(self):
        try:
            self.live.stop_if_idle()
        except Exception:
            pass
        st = state()
        attach_text = self.le_attach.text().strip()
        front_text = self.le_front.text().strip()
        defect_text = self.le_defect.text().strip()
        if attach_text:
            st.attachment_path = attach_text
            st.last_project_path = attach_text
        if front_text:
            st.front_attachment_path = front_text
            st.last_front_project_path = front_text
        if defect_text:
            st.defect_path = defect_text
            st.last_defect_project_path = defect_text
        save_state()
        super().accept()

    def reject(self):
        try:
            self.live.stop_if_idle()
        except Exception:
            pass
        super().reject()
