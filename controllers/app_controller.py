from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import QApplication

from controllers.hardware_controller import HardwareController
from controllers.live_camera_controller import LiveCameraController
from controllers.model_controller import ModelController
from ui.init_wizard import InitWizard
from ui.main_window import MainWindow


class AppController:
    """Top-level coordinator that wires controllers to the PyQt views."""

    def __init__(self):
        self.live = LiveCameraController()
        self.hardware = HardwareController()
        self.models = ModelController()
        self.window: Optional[MainWindow] = None

    def launch(self) -> Optional[MainWindow]:
        wizard = InitWizard(
            live_controller=self.live,
            hardware_controller=self.hardware,
            model_controller=self.models,
        )
        if wizard.exec_() != wizard.Accepted:
            return None
        self.window = MainWindow(self.live, self.hardware, self.models)
        self.window.show()
        return self.window
