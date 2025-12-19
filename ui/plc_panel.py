from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import (
    QWidget,
    QGroupBox,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QLabel,
)


class PlcPanel(QWidget):
    refresh_requested = pyqtSignal()
    connect_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        group = QGroupBox("Step 3 - Connect PLC (Modbus TCP)")
        root = QVBoxLayout(self)
        root.addWidget(group)

        v = QVBoxLayout(group)

        row = QHBoxLayout()
        self.host_combo = QComboBox()
        self.host_combo.setEditable(True)
        self.host_combo.setInsertPolicy(QComboBox.NoInsert)
        row.addWidget(self.host_combo, stretch=1)

        self.bt_refresh = QPushButton("Refresh")
        self.bt_refresh.clicked.connect(self.refresh_requested.emit)
        row.addWidget(self.bt_refresh)

        self.bt_connect = QPushButton("Connect")
        self._apply_connect_style(False)
        self.bt_connect.clicked.connect(self._on_connect_clicked)
        row.addWidget(self.bt_connect)

        v.addLayout(row)

        self.status = QLabel("Disconnected.")
        v.addWidget(self.status)

        self._connected = False

    def _on_connect_clicked(self):
        host = self.host_combo.currentText().strip()
        if host:
            self.connect_requested.emit(host)

    def set_hosts(self, hosts):
        self.host_combo.blockSignals(True)
        self.host_combo.clear()
        for h in hosts or []:
            self.host_combo.addItem(str(h))
        self.host_combo.blockSignals(False)

    def set_connected(self, connected: bool, endpoint: str = ""):
        self._connected = bool(connected)
        self.bt_connect.setText("Reconnect" if self._connected else "Connect")
        self._apply_connect_style(self._connected)
        self.set_status(endpoint or ("Connected." if self._connected else "Disconnected."))

    def set_status(self, text: str):
        self.status.setText(text or "")

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

