from typing import Optional
import numpy as np
from PyQt5.QtGui import QImage, QPixmap


def np_bgr_to_qpixmap(arr: np.ndarray) -> Optional[QPixmap]:
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except Exception:
        return None

    # Normalize to 3-channel BGR uint8.
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], 3, axis=2)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    else:
        return None

    if a.dtype != np.uint8:
        try:
            a = a.astype(np.uint8, copy=False)
        except Exception:
            return None

    bgr = np.ascontiguousarray(a)
    rgb = bgr[:, :, ::-1].copy()  # ensure contiguous and decoupled from source
    h, w = rgb.shape[:2]
    bytes_per_line = int(rgb.strides[0])
    # Copy the QImage so it doesn't reference a temporary numpy buffer.
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)
