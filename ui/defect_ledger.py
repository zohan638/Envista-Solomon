from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QSizePolicy,
)


class DefectLedger(QWidget):
    prev_requested = pyqtSignal()
    next_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Defect Ledger")
        v = QVBoxLayout(group)

        row = QHBoxLayout()

        self.top_label = QLabel("Annotated top not available.")
        self.top_label.setAlignment(Qt.AlignCenter)
        self.top_label.setStyleSheet("background: black;")
        self.top_label.setMinimumSize(200, 150)
        self.top_label.setScaledContents(False)
        self.top_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        row.addWidget(self.top_label, stretch=1)

        right_col = QVBoxLayout()
        self.front_label = QLabel("Front snapshot not available.")
        self.front_label.setAlignment(Qt.AlignCenter)
        self.front_label.setStyleSheet("background: black;")
        self.front_label.setMinimumSize(200, 150)
        self.front_label.setScaledContents(False)
        self.front_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(self.front_label, stretch=1)

        nav = QHBoxLayout()
        self.bt_prev = QPushButton("Previous")
        self.bt_prev.clicked.connect(self.prev_requested.emit)
        self.bt_next = QPushButton("Next")
        self.bt_next.clicked.connect(self.next_requested.emit)
        nav.addWidget(self.bt_prev)
        nav.addStretch(1)
        nav.addWidget(self.bt_next)
        right_col.addLayout(nav)

        row.addLayout(right_col, stretch=1)
        v.addLayout(row)

        layout.addWidget(group)

        self._top_base_pm: QPixmap = None
        self._front_base_pm: QPixmap = None

    # Rendering helpers
    def _apply_scaled_cover(self, label: QLabel, pm: QPixmap):
        if pm is None or pm.isNull():
            label.setPixmap(QPixmap())
            return
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        label.setPixmap(self._scale_and_crop(pm, target_w, target_h))

    def _scale_and_crop(self, pm: QPixmap, target_w: int, target_h: int) -> QPixmap:
        scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        x = max(0, (scaled.width() - target_w) // 2)
        y = max(0, (scaled.height() - target_h) // 2)
        return scaled.copy(x, y, target_w, target_h)

    def _render_top(self):
        if self._top_base_pm is None or self._top_base_pm.isNull():
            self.top_label.setText("Annotated top not available.")
            self.top_label.setPixmap(QPixmap())
            return
        self.top_label.setText("")
        self._apply_scaled_cover(self.top_label, self._top_base_pm)

    def _render_front(self):
        if self._front_base_pm is None or self._front_base_pm.isNull():
            self.front_label.setText("Front snapshot not available.")
            self.front_label.setPixmap(QPixmap())
            return
        self.front_label.setText("")
        self._apply_scaled_cover(self.front_label, self._front_base_pm)

    # Public API
    def set_top_pixmap(self, pm: QPixmap):
        self._top_base_pm = pm
        self._render_top()

    def set_front_pixmap(self, pm: QPixmap):
        self._front_base_pm = pm
        self._render_front()

    def set_images(self, *, top: QPixmap = None, front: QPixmap = None):
        if top is not None:
            self.set_top_pixmap(top)
        if front is not None:
            self.set_front_pixmap(front)

    def clear(self):
        self._top_base_pm = None
        self._front_base_pm = None
        self._render_top()
        self._render_front()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_top()
        self._render_front()
