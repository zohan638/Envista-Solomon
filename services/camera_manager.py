"""
Centralized camera capture that applies light settings before every shot and
flushes the camera buffer to avoid stale frames. Use this everywhere instead
of calling camera_service.capture() directly.
"""

from typing import Optional
import time
import cv2 as _cv2  # for optional saving
import threading

from . import camera_service as _cam
from . import light_controller as _lc
from .config import state as _state


_last_role: Optional[str] = None
_last_light_ip: Optional[str] = None
_last_light_ma: Optional[int] = None
_light_is_on: bool = False
_capture_lock = threading.RLock()

# Live preview should be responsive; downscale frames for display.
LIVE_PREVIEW_MAX_WIDTH = 960
LIVE_PREVIEW_MAX_HEIGHT = 540


def _apply_light(role: str) -> bool:
    """Best-effort: set the configured current on the single light (CH1) based
    on role ('Top' uses Top mA, 'Front' uses Front mA). No revert; each capture
    sets what it needs.
    """
    try:
        st = _state()
        ip = getattr(st, "light_ip", None)
        if not ip:
            return False
        enabled = getattr(st, "light_enabled", None)
        if enabled is False:
            return False
        _lc.configure(ip)
        # Single light only: always use channel 0 (CH1)
        target = int(
            getattr(
                st,
                "top_current_ma" if str(role).lower() == "top" else "front_current_ma",
                0,
            )
            or 0
        )
        global _last_light_ip, _last_light_ma, _light_is_on

        ip_s = str(ip)
        ma_i = int(target)
        changed = (ip_s != str(_last_light_ip)) or (ma_i != int(_last_light_ma if _last_light_ma is not None else -1))

        # Only touch the light when required (avoid spamming commands per frame).
        if changed or not _light_is_on:
            _lc.light_on(0)
            _light_is_on = True
        if changed:
            _lc.set_current_toggle(0, ma_i)
            _last_light_ip = ip_s
            _last_light_ma = ma_i
        return bool(changed)
    except Exception:
        # Do not break capture if light is unavailable
        return False


def capture(role: str, *, save_path: Optional[str] = None):
    """Capture a frame from the specified role ('Top' or 'Front').
    Steps:
      1) Apply light (CH1 only) for the role (verified internally by controller).
      2) Flush a couple frames from the backend queue (best-effort).
      3) Capture from the underlying camera_service.
      4) Optionally save to disk.
    """
    with _capture_lock:
        # Apply light (read-back ensure is inside light_controller)
        changed = _apply_light(role)

        # Optional dwell only when brightness changed
        try:
            st = _state()
            dwell_ms = int(getattr(st, "light_dwell_ms", 0) or 0)
        except Exception:
            dwell_ms = 0
        if changed and dwell_ms > 0:
            time.sleep(dwell_ms / 1000.0)

        # Flush any buffered frames, then capture
        try:
            from . import camera_service as _svc

            global _last_role
            deep = changed or (_last_role is not None and _last_role != role)
            _svc.flush(role, frames=(4 if deep else 2), timeout_ms=(80 if deep else 50))
            _last_role = role
        except Exception:
            pass

        frame = _cam.capture(role)

    try:
        print(f"[Camera] role={role} captured", flush=True)
    except Exception:
        pass

    if save_path:
        try:
            _cv2.imwrite(str(save_path), frame)
        except Exception:
            pass
    return frame


def capture_live(role: str, *, timeout_ms: int = 100, flush_frames: int = 0):
    """Low-latency capture intended for live UI preview.

    Does NOT touch the light controller (live preview must not change brightness).
    """
    with _capture_lock:
        try:
            from . import camera_service as _svc

            if flush_frames and int(flush_frames) > 0:
                _svc.flush(role, frames=max(0, int(flush_frames)), timeout_ms=25)
        except Exception:
            pass
        frame = _cam.capture(role, timeout_ms=int(timeout_ms))

    # Best-effort downscale for UI responsiveness.
    try:
        if frame is None or _cv2 is None:
            return frame
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return frame
        max_w = int(LIVE_PREVIEW_MAX_WIDTH)
        max_h = int(LIVE_PREVIEW_MAX_HEIGHT)
        if max_w <= 0 or max_h <= 0:
            return frame
        scale = min(max_w / float(w), max_h / float(h), 1.0)
        if scale >= 0.999:
            return frame
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        return _cv2.resize(frame, (new_w, new_h), interpolation=_cv2.INTER_AREA)
    except Exception:
        return frame


def is_connected(role: str) -> bool:
    return _cam.is_connected(role)


def connect(role: str, index: int) -> bool:
    return _cam.connect(role, index)


def disconnect(role: str) -> None:
    _cam.disconnect(role)


def enumerate_devices():
    return _cam.enumerate_devices()


def backend_name() -> str:
    return _cam.backend_name()


def diagnostics():
    return _cam.diagnostics()
