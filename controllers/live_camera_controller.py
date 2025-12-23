from __future__ import annotations

import concurrent.futures
import time
from typing import Dict, List, Optional, Tuple

from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from services import camera_manager, camera_service, debug_log
from services.config import save_state, state


class LiveCameraController(QObject):
    """
    Owns live camera streaming and connection lifecycle.

    Views subscribe to ``frame_ready`` / ``error_ready`` and drive their own
    rendering. The controller keeps scheduling logic and state persistence out
    of the UI layer.
    """

    frame_ready = pyqtSignal(str, int, object)  # role, generation, frame (np array)
    error_ready = pyqtSignal(str, int, str, str)  # role, generation, err_short, err_full
    devices_refreshed = pyqtSignal(list)

    def __init__(self, parent: Optional[QObject] = None, interval_ms: int = 50):
        super().__init__(parent)
        self._live_enabled = False
        self._live_closed = False
        self._live_gen: Dict[str, int] = {"Top": 0, "Front": 0}
        self._live_inflight: Dict[str, Optional[concurrent.futures.Future]] = {
            "Top": None,
            "Front": None,
        }
        self._live_err_ts: Dict[str, float] = {"Top": 0.0, "Front": 0.0}
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self._timer = QTimer(self)
        self._timer.setInterval(int(interval_ms))
        self._timer.timeout.connect(self._on_live_tick)

    # --- Connection helpers -------------------------------------------------
    def refresh_devices(self) -> List[dict]:
        devices = camera_service.enumerate_devices()
        try:
            self.devices_refreshed.emit(devices)
        except Exception:
            pass
        return devices

    def connect_camera(self, role: str, index: int) -> Tuple[bool, str]:
        """
        Connect a device index to a role ("Top"/"Front").
        Returns (connected, message). Prevents binding the same index to both roles.
        """
        role_norm = "Top" if str(role).lower().startswith("top") else "Front"
        other = "Front" if role_norm == "Top" else "Top"
        try:
            if camera_service.get_connected_index(other) == index:
                return False, f"Device {index} already bound to {other}."
        except Exception:
            pass

        connected = camera_service.connect(role_norm, index)
        msg = "Connected." if connected else "Connection failed."
        if connected:
            try:
                st = state()
                if role_norm == "Top":
                    st.camera_top_index = int(index)
                else:
                    st.camera_front_index = int(index)
                save_state()
            except Exception:
                pass
            self.start_live()
        else:
            self.bump_generation(role_norm)
        return connected, msg

    def disconnect_camera(self, role: str) -> None:
        role_norm = "Top" if str(role).lower().startswith("top") else "Front"
        camera_service.disconnect(role_norm)
        self.bump_generation(role_norm)
        self.stop_if_idle()

    def bump_generation(self, role: str) -> None:
        try:
            self._live_gen["Top" if role == "Top" else "Front"] += 1
        except Exception:
            pass

    # --- Live feed lifecycle ------------------------------------------------
    def start_live(self) -> None:
        if self._live_closed:
            return
        self._live_enabled = True
        try:
            if not self._timer.isActive():
                self._timer.start()
        except Exception:
            pass

    def stop_if_idle(self) -> None:
        try:
            if not camera_service.is_connected("Top") and not camera_service.is_connected("Front"):
                self._live_enabled = False
                self._timer.stop()
        except Exception:
            pass

    def stop_live(self) -> None:
        self._live_enabled = False
        try:
            self._timer.stop()
        except Exception:
            pass

    def shutdown(self) -> None:
        self._live_closed = True
        self.stop_live()
        try:
            self._executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            try:
                self._executor.shutdown(wait=False)
            except Exception:
                pass

    # --- Scheduling ---------------------------------------------------------
    def _on_live_tick(self) -> None:
        if self._live_closed or not self._live_enabled:
            return

        has_top = bool(camera_service.is_connected("Top"))
        has_front = bool(camera_service.is_connected("Front"))
        if not has_top and not has_front:
            self.stop_if_idle()
            return

        self._schedule_capture("Top")
        self._schedule_capture("Front")

    def _schedule_capture(self, role: str) -> None:
        if not camera_service.is_connected(role):
            return
        fut = self._live_inflight.get(role)
        if fut is not None and not fut.done():
            return
        gen = int(self._live_gen.get(role, 0) or 0)
        fut = self._executor.submit(camera_manager.capture_live, role)
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
                now = time.time()
                last = float(self._live_err_ts.get(role_inner, 0.0) or 0.0)
                if now - last >= 0.5:
                    self._live_err_ts[role_inner] = now
                    debug_log.log(f"live_capture error role={role_inner} err={err_short}")
                    self.error_ready.emit(role_inner, int(gen_inner), err_short, err)
                return
            if frame is None:
                debug_log.log(f"live_capture empty frame role={role_inner}")
                return
            try:
                self.frame_ready.emit(role_inner, int(gen_inner), frame)
            except Exception:
                pass

        try:
            fut.add_done_callback(_done)
        except Exception:
            pass
