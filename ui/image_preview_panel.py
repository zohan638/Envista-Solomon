from PyQt5.QtCore import Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPen
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QLabel,
    QSplitter,
)


class ImagePreviewPanel(QWidget):
    attachment_clicked = pyqtSignal(object)  # index clicked on the top overlay

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter, stretch=1)

        # Left: Attachment Overview
        self.group_attachment = QGroupBox("3. Attachment Overview")
        left_v = QVBoxLayout(self.group_attachment)
        self.original_label = QLabel()
        self.original_label.setStyleSheet("background: black; color: white;")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(200, 150)
        self.original_label.setScaledContents(False)
        left_v.addWidget(self.original_label)
        splitter.addWidget(self.group_attachment)

        # Right: Front Inspection
        self.group_front = QGroupBox("4. Front Inspection")
        right_v = QVBoxLayout(self.group_front)

        self.front_label = QLabel()
        self.front_label.setStyleSheet("background: black; color: white;")
        self.front_label.setAlignment(Qt.AlignCenter)
        self.front_label.setMinimumSize(200, 150)
        self.front_label.setScaledContents(False)
        right_v.addWidget(self.front_label, stretch=1)

        splitter.addWidget(self.group_front)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        # Keep base pixmaps (pre-overlay) for high-quality rescaling on resize
        self._original_base_pm = None
        self._front_base_pm = None
        self._front_overlay_enabled = True
        self._attachment_detections = []  # list of dicts: 'bounds','class','score', optional 'arrow'
        self._attachment_contour = None   # list of (x,y) points in original coords
        self._front_detections = []       # list of dicts for front view
        self._front_markers = []          # list of x positions (original pixel coords)
        self._draw_boxes = True
        self.original_label.installEventFilter(self)

    # Public helpers
    def _apply_scaled_cover(self, label: QLabel, pm: QPixmap):
        if pm is None or pm.isNull():
            label.setText("Failed to load image.")
            label.setPixmap(QPixmap())
            return
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        label.setPixmap(self._scale_and_crop(pm, target_w, target_h))

    def _scale_and_crop(self, pm: QPixmap, target_w: int, target_h: int) -> QPixmap:
        # First scale while preserving aspect ratio but ensuring we cover the target rect
        scaled = pm.scaled(target_w, target_h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        # Then center-crop to exact label size to avoid letterboxing
        x = max(0, (scaled.width() - target_w) // 2)
        y = max(0, (scaled.height() - target_h) // 2)
        return scaled.copy(x, y, target_w, target_h)

    def _set_pixmap(self, label: QLabel, path: str):
        pm = QPixmap(path)
        self._apply_scaled_cover(label, pm)

    def _render_original(self):
        if self._original_base_pm is not None:
            # Scale and crop to fit
            target_w = max(1, self.original_label.width())
            target_h = max(1, self.original_label.height())
            base = self._original_base_pm
            composed = self._scale_and_crop(base, target_w, target_h)
            # Apply overlay if we have detections or a tuned contour
            if self._attachment_detections or (self._attachment_contour is not None):
                composed = self._apply_attachment_overlay(base, composed, target_w, target_h)
            self.original_label.setPixmap(composed)

    def _render_front(self):
        if self._front_base_pm is None:
            return
        # Scale and crop first to match the label size exactly, then draw overlay
        w = max(1, self.front_label.width())
        h = max(1, self.front_label.height())
        base = self._front_base_pm
        composed = self._scale_and_crop(base, w, h)
        if self._front_overlay_enabled:
            composed = self._apply_front_overlay(base, composed, w, h)
        self.front_label.setPixmap(composed)

    def _apply_front_overlay(self, base: QPixmap, composed: QPixmap, target_w: int, target_h: int) -> QPixmap:
        # Draw red center crosshair and optional detection boxes on the composed pixmap
        if composed is None or composed.isNull():
            return composed
        result = QPixmap(composed)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Compute mapping like in _apply_attachment_overlay
        bw, bh = base.width(), base.height()
        if bw <= 0 or bh <= 0:
            return result
        sx = target_w / bw
        sy = target_h / bh
        s = max(sx, sy)
        scaled_w = int(bw * s)
        scaled_h = int(bh * s)
        off_x = max(0, (scaled_w - target_w) // 2)
        off_y = max(0, (scaled_h - target_h) // 2)

        # Red guides
        w = result.width()
        h = result.height()
        pen = QPen(QColor(220, 0, 0))
        pen.setWidth(1)
        painter.setPen(pen)
        # Center vertical line
        painter.drawLine(int(w * 0.5), 0, int(w * 0.5), h)
        # Center horizontal line
        painter.drawLine(0, int(h * 0.5), w, int(h * 0.5))
        # Center circle
        r = int(min(w, h) * 0.06)
        cx = w // 2
        cy = h // 2
        painter.drawEllipse(cx - r, cy - r, 2 * r, 2 * r)
        # Optional detection rectangles for front view
        if self._front_detections:
            from PyQt5.QtGui import QFont
            box_pen = QPen(QColor(0, 200, 83))
            box_pen.setWidth(2)
            painter.setPen(box_pen)
            font = QFont(); font.setPointSize(9); painter.setFont(font)
            for d in self._front_detections:
                try:
                    b = d.get("bounds") or d.get("rect")
                    if not b:
                        continue
                    x, y, ww, hh = b
                    dx = int(x * s - off_x)
                    dy = int(y * s - off_y)
                    dw = int(ww * s)
                    dh = int(hh * s)
                    if dx + dw < 0 or dy + dh < 0 or dx > target_w or dy > target_h:
                        continue
                    painter.drawRect(dx, dy, dw, dh)
                    label = str(d.get("class", ""))
                    score = d.get("score")
                    if score is not None:
                        try:
                            label += f" {float(score):.2f}"
                        except Exception:
                            pass
                    if label:
                        metrics = painter.fontMetrics()
                        tw = metrics.width(label) + 6
                        th = metrics.height() + 4
                        painter.fillRect(dx, max(0, dy - th), tw, th, QColor(0, 200, 83, 180))
                        painter.setPen(QColor(255, 255, 255))
                        painter.drawText(dx + 3, dy - 4, label)
                        painter.setPen(box_pen)
                except Exception:
                    continue
        painter.end()
        # Draw blue markers (scaled x positions) on top of result
        if self._front_markers:
            painter = QPainter(result)
            painter.setRenderHint(QPainter.Antialiasing, True)
            # same mapping s/off_x/off_y computed above
            bw, bh = base.width(), base.height()
            sx = target_w / bw; sy = target_h / bh; s = max(sx, sy)
            scaled_w = int(bw * s); scaled_h = int(bh * s)
            off_x = max(0, (scaled_w - target_w) // 2)
            # place on horizontal midline
            dot_pen = QPen(QColor(0, 122, 204)); dot_pen.setWidth(2)
            painter.setPen(dot_pen)
            from PyQt5.QtGui import QBrush
            painter.setBrush(QColor(0, 122, 204))
            for x in self._front_markers:
                try:
                    dx = int(x * s - off_x)
                    dy = target_h // 2
                    painter.drawEllipse(dx - 5, dy - 5, 10, 10)
                except Exception:
                    continue
            painter.end()
        return result

    def _apply_attachment_overlay(self, base: QPixmap, composed: QPixmap, target_w: int, target_h: int) -> QPixmap:
        # Draw detection markers (filled circles) on the already scaled+cropped image.
        if composed is None or composed.isNull():
            return composed
        result = QPixmap(composed)
        from PyQt5.QtGui import QFont, QBrush
        painter = QPainter(result)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # Compute scale and crop offsets used by _scale_and_crop
        bw, bh = base.width(), base.height()
        if bw <= 0 or bh <= 0:
            return result
        sx = target_w / bw
        sy = target_h / bh
        s = max(sx, sy)
        scaled_w = int(bw * s)
        scaled_h = int(bh * s)
        off_x = max(0, (scaled_w - target_w) // 2)
        off_y = max(0, (scaled_h - target_h) // 2)

        def _color_for_state(state: str) -> QColor:
            st = (state or "").lower()
            if st in {"fail", "defect"}:
                return QColor(220, 20, 60, 150)  # red, semi-transparent
            if st == "ok":
                return QColor(46, 125, 50, 140)  # green, semi-transparent
            return QColor(255, 215, 0, 140)  # yellow default

        # Draw each detection as a filled circle with label + heading
        for d in self._attachment_detections:
            try:
                b = d.get("bounds") or d.get("rect")
                if not b:
                    continue
                x, y, w, h = b
                # Map to composed coordinates
                dx = int(x * s - off_x)
                dy = int(y * s - off_y)
                dw = int(w * s)
                dh = int(h * s)
                # Skip if fully outside
                if dx + dw < 0 or dy + dh < 0 or dx > target_w or dy > target_h:
                    continue
                cx = dx + dw // 2
                cy = dy + dh // 2
                radius = 18  # uniform size for all markers
                state = d.get("defect_state")
                fill = _color_for_state(state)
                border = QColor(fill)
                border.setAlpha(220)
                painter.setBrush(QBrush(fill))
                pen = QPen(border)
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawEllipse(cx - radius, cy - radius, radius * 2, radius * 2)

                # Text: index centered
                font = QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                idx_val = d.get("index")
                label = str(idx_val) if idx_val is not None else ""
                if label:
                    metrics = painter.fontMetrics()
                    tw = metrics.width(label)
                    th = metrics.height()
                    painter.setPen(QColor(0, 0, 0))
                    painter.drawText(cx - tw // 2, cy + th // 4, label)
            except Exception:
                continue
        # Optional tuned contour polyline
        try:
            cnt = self._attachment_contour
            if cnt is not None:
                blue = QPen(QColor(0, 122, 204))
                blue.setWidth(2)
                painter.setPen(blue)
                n = len(cnt)
                for i in range(max(0, n - 1)):
                    try:
                        px, py = cnt[i]
                        qx, qy = cnt[i + 1]
                        x1 = int(px * s - off_x); y1 = int(py * s - off_y)
                        x2 = int(qx * s - off_x); y2 = int(qy * s - off_y)
                        painter.drawLine(x1, y1, x2, y2)
                    except Exception:
                        pass
                if n >= 3:
                    try:
                        px, py = cnt[0]; qx, qy = cnt[-1]
                        x1 = int(px * s - off_x); y1 = int(py * s - off_y)
                        x2 = int(qx * s - off_x); y2 = int(qy * s - off_y)
                        painter.drawLine(x1, y1, x2, y2)
                    except Exception:
                        pass
        except Exception:
            pass
        painter.end()
        return result

    def _map_label_to_original(self, label: QLabel, pos):
        """Map a click position in label coords to original image coords."""
        if self._original_base_pm is None or self._original_base_pm.isNull():
            return None, None
        bw, bh = self._original_base_pm.width(), self._original_base_pm.height()
        if bw <= 0 or bh <= 0:
            return None, None
        target_w = max(1, label.width())
        target_h = max(1, label.height())
        sx = target_w / bw
        sy = target_h / bh
        s = max(sx, sy)
        scaled_w = bw * s
        scaled_h = bh * s
        off_x = max(0, (scaled_w - target_w) / 2.0)
        off_y = max(0, (scaled_h - target_h) / 2.0)
        x_orig = (pos.x() + off_x) / s
        y_orig = (pos.y() + off_y) / s
        return x_orig, y_orig

    def _emit_attachment_click(self, x_orig, y_orig):
        if not self._attachment_detections:
            return
        best_idx = None
        best_dist2 = None
        for d in self._attachment_detections:
            try:
                idx = d.get("index")
                b = d.get("bounds")
                if not b or idx is None:
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
        try:
            if best_idx is not None:
                # Require a reasonable proximity (~60 px in original coords)
                if best_dist2 is None or best_dist2 <= (60 * 60):
                    self.attachment_clicked.emit(best_idx)
        except Exception:
            pass

    def eventFilter(self, obj, event):
        if obj == self.original_label and event.type() == QEvent.MouseButtonPress:
            x, y = self._map_label_to_original(self.original_label, event.pos())
            if x is not None and y is not None:
                self._emit_attachment_click(x, y)
            return False
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Rescale images on resize to keep zoom behavior similar to PictureBox Zoom
        self._render_original()
        self._render_front()

    def set_original_image(self, path: str):
        self._original_base_pm = QPixmap(path)
        self._render_original()

    def set_front_preview_image(self, path: str):
        self._front_base_pm = QPixmap(path)
        self._render_front()

    # New helpers for numpy images
    def set_original_np(self, pixmap: QPixmap):
        self._original_base_pm = pixmap
        self._render_original()

    def set_front_np(self, pixmap: QPixmap):
        self._front_base_pm = pixmap
        self._render_front()

    def set_overlay_enabled(self, enabled: bool):
        self._front_overlay_enabled = bool(enabled)
        self._render_front()

    def set_attachment_detections(self, detections):
        # Expect list of dicts with 'bounds': (x,y,w,h) and optional class/score
        self._attachment_detections = detections or []
        self._render_original()

    def set_draw_boxes(self, enabled: bool):
        self._draw_boxes = bool(enabled)
        self._render_original()

    def set_front_detections(self, detections):
        self._front_detections = detections or []
        self._render_front()

    def set_front_markers(self, xs):
        # xs: iterable of x positions in original front-image pixels
        try:
            self._front_markers = [int(x) for x in (xs or [])]
        except Exception:
            self._front_markers = []
        self._render_front()

    def set_attachment_contour(self, contour_points):
        # Accept list/ndarray of (x,y) points in original image coordinates
        try:
            self._attachment_contour = list(contour_points) if contour_points is not None else None
        except Exception:
            self._attachment_contour = None
        self._render_original()

    # Export helpers for saving composed attachment view (with overlays)
    def capture_attachment_view(self):
        if self._original_base_pm is None:
            return None
        target_w = max(1, self.original_label.width())
        target_h = max(1, self.original_label.height())
        base = self._original_base_pm
        composed = self._scale_and_crop(base, target_w, target_h)
        if self._attachment_detections or (self._attachment_contour is not None):
            composed = self._apply_attachment_overlay(base, composed, target_w, target_h)
        return composed

    def capture_attachment_view_fullres(self):
        """Capture annotated attachment view at native resolution (no cropping/letterboxing)."""
        if self._original_base_pm is None:
            return None
        base = self._original_base_pm
        composed = QPixmap(base)
        if self._attachment_detections or (self._attachment_contour is not None):
            composed = self._apply_attachment_overlay(base, composed, base.width(), base.height())
        return composed

    def save_attachment_view(self, path: str) -> bool:
        pm = self.capture_attachment_view()
        if pm is None or pm.isNull():
            return False
        try:
            # Infer format from suffix; default to PNG
            suffix = (path.rsplit('.', 1)[-1] or 'png').upper() if '.' in path else 'PNG'
            return bool(pm.save(path, suffix))
        except Exception:
            return False

    def save_attachment_view_fullres(self, path: str) -> bool:
        pm = self.capture_attachment_view_fullres()
        if pm is None or pm.isNull():
            return False
        try:
            suffix = (path.rsplit('.', 1)[-1] or 'png').upper() if '.' in path else 'PNG'
            return bool(pm.save(path, suffix))
        except Exception:
            return False

    def capture_front_view(self):
        if self._front_base_pm is None:
            return None
        target_w = max(1, self.front_label.width())
        target_h = max(1, self.front_label.height())
        base = self._front_base_pm
        composed = self._scale_and_crop(base, target_w, target_h)
        if self._front_overlay_enabled:
            composed = self._apply_front_overlay(base, composed, target_w, target_h)
        return composed

    def render_attachment_overlay(self, base_pixmap: QPixmap, detections, contour=None):
        """
        Render an annotated attachment view using the provided base pixmap and detections
        without mutating the UI state or labels.
        """
        if base_pixmap is None or base_pixmap.isNull():
            return None
        prev_det = self._attachment_detections
        prev_cnt = self._attachment_contour
        try:
            self._attachment_detections = detections or []
            if contour is not None:
                self._attachment_contour = contour
            return self._apply_attachment_overlay(base_pixmap, base_pixmap, base_pixmap.width(), base_pixmap.height())
        except Exception:
            return None
        finally:
            self._attachment_detections = prev_det
            self._attachment_contour = prev_cnt

    def save_front_view(self, path: str) -> bool:
        pm = self.capture_front_view()
        if pm is None or pm.isNull():
            return False
        try:
            suffix = (path.rsplit('.', 1)[-1] or 'png').upper() if '.' in path else 'PNG'
            return bool(pm.save(path, suffix))
        except Exception:
            return False
