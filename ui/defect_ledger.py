from PyQt5.QtCore import Qt, QEvent, pyqtSignal
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
    selection_changed = pyqtSignal(object)  # emits index when user navigates

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Defect Ledger")
        v = QVBoxLayout(group)

        row = QHBoxLayout()

        self.top_label = QLabel("Annotated top not available.")
        self.top_label.setAlignment(Qt.AlignCenter)
        self.top_label.setStyleSheet("background: black; color: white;")
        self.top_label.setMinimumSize(200, 150)
        self.top_label.setScaledContents(False)
        self.top_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.top_label.installEventFilter(self)
        row.addWidget(self.top_label, stretch=1)

        right_col = QVBoxLayout()
        self.front_label = QLabel("Front snapshot not available.")
        self.front_label.setAlignment(Qt.AlignCenter)
        self.front_label.setStyleSheet("background: black; color: white;")
        self.front_label.setMinimumSize(200, 150)
        self.front_label.setScaledContents(False)
        self.front_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_col.addWidget(self.front_label, stretch=1)

        nav = QHBoxLayout()
        self.bt_prev = QPushButton("Previous")
        self.bt_prev.clicked.connect(self.prev_requested.emit)
        self.lbl_index = QLabel("0 / 0")
        self.lbl_index.setAlignment(Qt.AlignCenter)
        self.bt_next = QPushButton("Next")
        self.bt_next.clicked.connect(self.next_requested.emit)
        nav.addWidget(self.bt_prev)
        nav.addWidget(self.lbl_index)
        nav.addWidget(self.bt_next)
        right_col.addLayout(nav)

        row.addLayout(right_col, stretch=1)
        v.addLayout(row)

        layout.addWidget(group)

        self._top_base_pm: QPixmap = None
        self._front_items = {}  # idx -> (QPixmap, message)
        self._order = []  # sorted indices for navigation
        self._current_idx = None
        self._total = 0
        self._front_base_pm: QPixmap = None
        self._front_msg: str = None
        self._top_detections = []  # copy of detections for click-mapping
        self._top_image_size = None  # (w, h) of the annotated top pixmap

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
            msg = self._front_msg or "Front snapshot not available."
            self.front_label.setText(msg)
            self.front_label.setPixmap(QPixmap())
            return
        self.front_label.setText("")
        self._apply_scaled_cover(self.front_label, self._front_base_pm)

    def _update_index_label(self):
        total = max(int(self._total or 0), len(self._order or []))
        try:
            cur_val = int(self._current_idx) if self._current_idx is not None else 0
        except Exception:
            cur_val = self._current_idx or 0
        self.lbl_index.setText(f"{cur_val} / {total}")
        self.bt_prev.setEnabled(total > 0)
        self.bt_next.setEnabled(total > 0)

    def _set_current_front_pixmap(self, idx: int):
        item = self._front_items.get(idx)
        if isinstance(item, tuple):
            pm, msg = item
        else:
            pm, msg = (item if item is not None else None), None
        if pm is None or (hasattr(pm, "isNull") and pm.isNull()):
            pm = None
            if not msg:
                msg = "No image for this attachment."
        self._front_base_pm = pm
        self._front_msg = msg
        self._render_front()
        self._current_idx = idx
        self._update_index_label()
        try:
            self.selection_changed.emit(idx)
        except Exception:
            pass

    # Public API
    def set_top_pixmap(self, pm: QPixmap, detections=None, image_size=None):
        self._top_base_pm = pm
        try:
            self._top_detections = list(detections or [])
        except Exception:
            self._top_detections = []
        if image_size:
            try:
                w, h = image_size
                self._top_image_size = (int(w), int(h))
            except Exception:
                self._top_image_size = None
        elif pm is not None and not pm.isNull():
            self._top_image_size = (pm.width(), pm.height())
        else:
            self._top_image_size = None
        self._render_top()

    def set_total(self, total: int):
        try:
            self._total = max(0, int(total))
        except Exception:
            self._total = 0
        self._update_index_label()

    def set_front_for_index(self, idx: int, pm: QPixmap, message: str = None):
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = idx
        self._front_items[idx_int] = (pm, message)
        if idx_int not in self._order:
            self._order.append(idx_int)
            try:
                self._order = sorted(dict.fromkeys(self._order))
            except Exception:
                pass
        self._set_current_front_pixmap(idx_int)

    def set_front_pixmap(self, pm: QPixmap, idx=None, message: str = None):
        """
        Update the front panel with a pixmap (optionally tied to an attachment index).
        If idx is provided, the image is stored and becomes the current selection.
        """
        if idx is None:
            self._front_base_pm = pm
            self._front_msg = message
            self._render_front()
            self._update_index_label()
            return
        self.set_front_for_index(idx, pm, message=message)

    def clear(self):
        self._top_base_pm = None
        self._front_base_pm = None
        self._front_msg = None
        self._top_detections = []
        self._top_image_size = None
        self._front_items = {}
        self._order = []
        self._current_idx = None
        self._total = 0
        self._render_top()
        self._render_front()
        self._update_index_label()

    def clear_items(self, total: int = 0):
        self._front_items = {}
        self._order = []
        self._current_idx = None
        self._front_base_pm = None
        self._front_msg = None
        self._top_detections = []
        self._top_image_size = None
        self.set_total(total)
        self._render_front()

    def set_index_order(self, indices):
        try:
            numeric = []
            for i in indices or []:
                try:
                    numeric.append(int(i))
                except Exception:
                    pass
            # Sort numerically for stable navigation order
            self._order = sorted(dict.fromkeys(numeric))
        except Exception:
            self._order = []
        if self._order:
            self._current_idx = self._order[0]
        self._update_index_label()
        self._render_front()

    def set_current_index(self, idx):
        if idx is None:
            return
        try:
            idx_int = int(idx)
        except Exception:
            idx_int = idx
        # Keep order for navigation even if no image yet
        if idx_int not in self._order:
            self._order.append(idx_int)
            try:
                # Keep navigation order stable and ascending
                self._order = sorted(dict.fromkeys(self._order))
            except Exception:
                pass
        self._set_current_front_pixmap(idx_int)

    def select_next(self):
        if not self._order:
            return
        if self._current_idx not in self._order:
            self._set_current_front_pixmap(self._order[0])
            return
        pos = self._order.index(self._current_idx)
        nxt = self._order[(pos + 1) % len(self._order)]
        self._set_current_front_pixmap(nxt)

    def select_prev(self):
        if not self._order:
            return
        if self._current_idx not in self._order:
            self._set_current_front_pixmap(self._order[0])
            return
        pos = self._order.index(self._current_idx)
        prv = self._order[(pos - 1) % len(self._order)]
        self._set_current_front_pixmap(prv)

    def _map_top_label_to_original(self, pos):
        """Map a click on the top label back to original top-image coordinates."""
        if self._top_image_size is None:
            if self._top_base_pm is None or self._top_base_pm.isNull():
                return None, None
            self._top_image_size = (self._top_base_pm.width(), self._top_base_pm.height())
        try:
            bw, bh = self._top_image_size
        except Exception:
            return None, None
        if not bw or not bh:
            return None, None
        target_w = max(1, self.top_label.width())
        target_h = max(1, self.top_label.height())
        sx = target_w / bw
        sy = target_h / bh
        s = max(sx, sy)
        scaled_w = bw * s
        scaled_h = bh * s
        off_x = max(0.0, (scaled_w - target_w) / 2.0)
        off_y = max(0.0, (scaled_h - target_h) / 2.0)
        x_orig = (pos.x() + off_x) / s
        y_orig = (pos.y() + off_y) / s
        return x_orig, y_orig

    def _emit_top_click(self, x_orig, y_orig):
        if not self._top_detections:
            return
        best_idx = None
        best_dist2 = None
        for d in self._top_detections:
            try:
                idx = d.get("index")
                if idx is None:
                    continue
                if "det_center" in d and d.get("det_center"):
                    cx, cy = d.get("det_center")
                else:
                    b = d.get("bounds") or d.get("rect")
                    if not b:
                        continue
                    cx = b[0] + b[2] / 2.0
                    cy = b[1] + b[3] / 2.0
                dx = cx - x_orig
                dy = cy - y_orig
                dist2 = dx * dx + dy * dy
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    best_idx = idx
            except Exception:
                continue
        if best_idx is None:
            return
        # Require a reasonable proximity (~60 px in original coords)
        if best_dist2 is None or best_dist2 <= (60 * 60):
            self.set_current_index(best_idx)

    def eventFilter(self, obj, event):
        if obj == self.top_label and event.type() == QEvent.MouseButtonPress:
            x, y = self._map_top_label_to_original(event.pos())
            if x is not None and y is not None:
                self._emit_top_click(x, y)
            return False
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_top()
        self._render_front()
