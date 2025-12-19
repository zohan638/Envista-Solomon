from typing import Optional

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QComboBox,
    QPushButton,
    QLabel,
    QSpinBox,
)

_ACT_STATE_NAMES = {
    10: "UNCAL",
    20: "IDLE",
    30: "CAL_SEEK_LEFT",
    40: "CAL_SEEK_RIGHT_COUNT",
    60: "MOVE",
    70: "JOG",
    90: "FAULT",
}

_ACT_FAULT_NAMES = {
    0: "OK",
    1: "Drive alarm",
    2: "Limit fault",
    3: "Not calibrated",
    4: "Timeout",
}


class LinearAxisPanel(QWidget):
    refresh_requested = pyqtSignal()
    connect_requested = pyqtSignal(str)
    calibrate_requested = pyqtSignal()
    home_requested = pyqtSignal(int)
    goto_requested = pyqtSignal(int)
    port_selected = pyqtSignal(str)
    home_set_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        group = QGroupBox("Linear Axis (Front Camera)")
        root = QVBoxLayout(self)
        root.addWidget(group)

        v = QVBoxLayout(group)

        # Connection row
        row = QHBoxLayout()
        self.port_combo = QComboBox()
        self.port_combo.setEditable(True)
        self.port_combo.currentTextChanged.connect(lambda s: self.port_selected.emit(s))
        row.addWidget(self.port_combo, stretch=1)

        self.bt_refresh = QPushButton("Refresh")
        self.bt_refresh.clicked.connect(self.refresh_requested.emit)
        row.addWidget(self.bt_refresh)

        self.bt_connect = QPushButton("Connect")
        self._apply_connect_style(False)
        self.bt_connect.clicked.connect(self._on_connect)
        row.addWidget(self.bt_connect)

        v.addLayout(row)

        # Calibration / home row
        cal_row = QHBoxLayout()
        self.bt_calibrate = QPushButton("Calibrate")
        self.bt_calibrate.clicked.connect(self.calibrate_requested.emit)
        cal_row.addWidget(self.bt_calibrate)

        self.bt_home = QPushButton("Home (0 steps)")
        self.bt_home.clicked.connect(self._on_home)
        cal_row.addWidget(self.bt_home)

        v.addLayout(cal_row)

        # Live PLC status (read-only)
        grid = QGridLayout()

        grid.addWidget(QLabel("Live pos (steps):"), 0, 0)
        self.lbl_live_pos = QLabel("-")
        grid.addWidget(self.lbl_live_pos, 0, 1)

        grid.addWidget(QLabel("Target (steps):"), 0, 2)
        self.lbl_target = QLabel("-")
        grid.addWidget(self.lbl_target, 0, 3)

        grid.addWidget(QLabel("State:"), 1, 0)
        self.lbl_state = QLabel("-")
        grid.addWidget(self.lbl_state, 1, 1)

        grid.addWidget(QLabel("In motion:"), 1, 2)
        self.lbl_in_motion = QLabel("-")
        grid.addWidget(self.lbl_in_motion, 1, 3)

        grid.addWidget(QLabel("Calibrated:"), 2, 0)
        self.lbl_cal = QLabel("-")
        grid.addWidget(self.lbl_cal, 2, 1)

        grid.addWidget(QLabel("Total steps:"), 2, 2)
        self.lbl_total = QLabel("-")
        grid.addWidget(self.lbl_total, 2, 3)

        grid.addWidget(QLabel("Fault:"), 3, 0)
        self.lbl_fault = QLabel("-")
        grid.addWidget(self.lbl_fault, 3, 1)

        v.addLayout(grid)

        # Goto row (user input; never overwritten by live polling)
        goto_row = QHBoxLayout()
        goto_row.addWidget(QLabel("Goto (steps):"))
        self.goto_spin = QSpinBox()
        self.goto_spin.setRange(0, 10_000_000)
        self.goto_spin.setSingleStep(100)
        self.goto_spin.setValue(0)
        goto_row.addWidget(self.goto_spin)

        self.bt_goto = QPushButton("Go")
        self.bt_goto.clicked.connect(self._on_goto)
        goto_row.addWidget(self.bt_goto)

        v.addLayout(goto_row)

        # Home setter uses live position (not the goto input)
        home_row = QHBoxLayout()
        self.bt_set_home = QPushButton("Set Home to Current Position")
        self.bt_set_home.clicked.connect(self._on_set_home)
        home_row.addWidget(self.bt_set_home)
        v.addLayout(home_row)

        self.status = QLabel("Disconnected.")
        v.addWidget(self.status)

        self._connected = False
        self._calibrated = False
        self._ready = False
        self._calibrating = False
        self._home_steps = 0
        self._live_position_steps: Optional[int] = None
        self._update_enabled()

    def _on_connect(self):
        port = self.port_combo.currentText().strip()
        if port:
            self.connect_requested.emit(port)

    def _on_home(self):
        self.home_requested.emit(int(self._home_steps))

    def _on_goto(self):
        self.goto_requested.emit(int(self.goto_spin.value()))

    def _on_set_home(self):
        steps = int(self._live_position_steps or 0)
        self.set_home_steps(steps)
        self.home_set_requested.emit(int(steps))

    def set_calibrating(self, calibrating: bool):
        self._calibrating = bool(calibrating)
        if self._calibrating:
            self.set_status("Calibrating...")
        self._update_enabled()
        if not self._calibrating and self._calibrated:
            self.set_status(f"Calibrated. Live position: {int(self._live_position_steps or 0)} steps.")

    def set_ports(self, ports):
        self.port_combo.blockSignals(True)
        self.port_combo.clear()
        for p in ports:
            self.port_combo.addItem(p)
        self.port_combo.blockSignals(False)

    def set_connected(self, connected: bool, port: Optional[str] = None):
        self._connected = connected
        if not connected:
            self._ready = False
            self._calibrated = False
            self._calibrating = False
            self._live_position_steps = None
            try:
                self.lbl_live_pos.setText("-")
                self.lbl_target.setText("-")
                self.lbl_state.setText("-")
                self.lbl_in_motion.setText("-")
                self.lbl_cal.setText("-")
                self.lbl_total.setText("-")
                self.lbl_fault.setText("-")
            except Exception:
                pass
        else:
            self._ready = False
        self._update_enabled()
        self.bt_connect.setText("Reconnect" if connected else "Connect")
        self._apply_connect_style(connected)
        if connected:
            self.set_status(f"Connected ({port or ''}). Waiting for controller...")
        else:
            self.set_status("Disconnected.")

    def set_calibrated(self, calibrated: bool, position_steps: Optional[int] = None, total_steps: Optional[int] = None):
        self._calibrated = calibrated
        self._calibrating = False
        if calibrated:
            self._ready = True
            if isinstance(total_steps, int) and total_steps > 0:
                try:
                    self.goto_spin.setMaximum(int(total_steps))
                except Exception:
                    pass
        self._update_enabled()
        if calibrated:
            if position_steps is not None:
                try:
                    self.set_position(int(position_steps))
                except Exception:
                    pass
            self.set_status(f"Calibrated. Live position: {int(self._live_position_steps or 0)} steps.")
        else:
            self.set_status("Connected but not calibrated.")

    def set_position(self, position_steps: int):
        try:
            p = int(position_steps)
            self._live_position_steps = p
            self.lbl_live_pos.setText(str(p))
        except Exception:
            pass

    def set_ready(self, ready: bool):
        self._ready = bool(ready)
        self._update_enabled()
        if self._connected:
            if self._ready:
                self.set_status(f"Controller ready. Live position: {int(self._live_position_steps or 0)} steps.")
            else:
                self.set_status("Connected. Waiting for controller...")

    def set_status(self, text: str):
        self.status.setText(text or "")

    def set_home_steps(self, steps: int):
        try:
            self._home_steps = int(steps)
            self._update_home_button_text(int(steps))
        except Exception:
            pass

    def home_steps(self) -> int:
        return int(self._home_steps)

    def is_calibrating(self) -> bool:
        return bool(self._calibrating)

    def is_ready(self) -> bool:
        return bool(self._ready)

    def _update_enabled(self):
        connected = self._connected
        busy = self._calibrating

        # Do not allow changing ports or reconnecting while a calibration is running
        self.port_combo.setEnabled(not connected and not busy)
        self.bt_refresh.setEnabled(not connected and not busy)
        self.bt_connect.setEnabled(not busy)

        # Calibrate requires controller ready, connection, not calibrated, and no ongoing calibration
        can_calibrate = connected and self._ready and (not self._calibrated) and (not busy)
        self.bt_calibrate.setEnabled(can_calibrate)

        # Home/Go/position require calibration and should stay disabled while calibrating
        can_move = connected and self._calibrated and not busy
        self.bt_home.setEnabled(can_move)
        self.goto_spin.setEnabled(can_move)
        self.bt_goto.setEnabled(can_move)
        # Home config requires controller ready (to avoid pre-banner clicks)
        can_configure_home = connected and self._ready and not busy
        self.bt_set_home.setEnabled(can_configure_home)
        # Keep connect button style in sync after enable/disable changes
        self._apply_connect_style(connected)

        # Keep calibrate button text meaningful
        if self._calibrating:
            self.bt_calibrate.setText("Calibrating...")
        elif self._calibrated:
            self.bt_calibrate.setText("Calibrated")
        else:
            self.bt_calibrate.setText("Calibrate")

        # Calibrate button color: red while calibrating, green when calibrated, default otherwise
        if self._calibrating:
            self.bt_calibrate.setStyleSheet(
                "QPushButton {"
                " background-color: #c62828; color: white;"
                " border: 2px solid #c62828; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #d32f2f; }"
            )
        elif self._calibrated:
            self.bt_calibrate.setStyleSheet(
                "QPushButton {"
                " background-color: #2e7d32; color: white;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #388e3c; }"
            )
        else:
            self.bt_calibrate.setStyleSheet(
                "QPushButton {"
                " background: transparent; color: #2e7d32;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(46,125,50,0.08); }"
            )

    def _apply_connect_style(self, connected: bool):
        if connected:
            self.bt_connect.setStyleSheet(
                "QPushButton {"
                " background-color: #2e7d32; color: white;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: #388e3c; }"
            )
        else:
            self.bt_connect.setStyleSheet(
                "QPushButton {"
                " background: transparent; color: #2e7d32;"
                " border: 2px solid #2e7d32; padding: 6px 10px; font-weight: 600;"
                "}"
                "QPushButton:hover { background-color: rgba(46,125,50,0.08); }"
            )

    def _update_home_button_text(self, steps: int):
        try:
            self.bt_home.setText(f"Home ({int(steps)} steps)")
        except Exception:
            pass

    def set_plc_axis_snapshot(
        self,
        *,
        position_steps: Optional[int] = None,
        target_steps: Optional[int] = None,
        in_motion: Optional[bool] = None,
        act_state: Optional[int] = None,
        act_fault_code: Optional[int] = None,
        calibrated: Optional[bool] = None,
        total_steps: Optional[int] = None,
    ):
        """
        Update read-only PLC-derived labels. This must never overwrite the user's goto input.
        """
        try:
            if position_steps is not None:
                self.set_position(int(position_steps))
        except Exception:
            pass

        try:
            if target_steps is not None:
                self.lbl_target.setText(str(int(target_steps)))
        except Exception:
            pass

        try:
            if in_motion is not None:
                self.lbl_in_motion.setText("YES" if bool(in_motion) else "NO")
        except Exception:
            pass

        try:
            if act_state is not None:
                s = int(act_state)
                name = _ACT_STATE_NAMES.get(s, "UNKNOWN")
                self.lbl_state.setText(f"{name} ({s})")
        except Exception:
            pass

        try:
            if calibrated is not None:
                self.lbl_cal.setText("YES" if bool(calibrated) else "NO")
        except Exception:
            pass

        try:
            if total_steps is not None and int(total_steps) > 0:
                self.lbl_total.setText(str(int(total_steps)))
                try:
                    self.goto_spin.setMaximum(int(total_steps))
                except Exception:
                    pass
            elif total_steps is not None:
                self.lbl_total.setText("-")
        except Exception:
            pass

        try:
            if act_fault_code is not None:
                fc = int(act_fault_code)
                name = _ACT_FAULT_NAMES.get(fc, "Unknown")
                self.lbl_fault.setText(f"{name} ({fc})")
        except Exception:
            pass
