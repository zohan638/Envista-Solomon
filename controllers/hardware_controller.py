from __future__ import annotations

import threading
from typing import List, Optional, Tuple

from PyQt5.QtCore import QObject, pyqtSignal

from services import linear_axis_service, plc_service, turntable_service, debug_log
from services.config import save_state, state


class HardwareController(QObject):
    """
    Encapsulates PLC-driven motion: turntable and linear axis helpers.

    Emits Qt signals for UI updates so the view layer stays presentation-only.
    """

    turntable_message = pyqtSignal(str)
    turntable_status = pyqtSignal(str)
    plc_snapshot = pyqtSignal(object)
    axis_log = pyqtSignal(str)
    axis_ready = pyqtSignal(bool)
    axis_calibrating = pyqtSignal(bool)
    axis_calibrated = pyqtSignal(bool, object, object)
    axis_position = pyqtSignal(object)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._tt_listener = self._on_turntable_message
        turntable_service.add_listener(self._tt_listener)
        try:
            plc_service.add_status_listener(self._emit_plc_snapshot)
        except Exception:
            pass

    # --- Turntable / PLC ----------------------------------------------------
    def refresh_turntable_devices(self) -> List[str]:
        return turntable_service.refresh_devices()

    def connect_turntable(self, port: str) -> Tuple[bool, str]:
        debug_log.log(f"turntable_connect start port={port}")
        ok = turntable_service.connect(port)
        msg = "Connected." if ok else "Connection failed."
        if ok:
            try:
                st = state()
                st.plc_host = str(port).split(":", 1)[0].strip() or str(port).strip()
                save_state()
            except Exception:
                pass
            try:
                ep = turntable_service.port_name() or port
                self.turntable_status.emit(ep)
            except Exception:
                pass
        debug_log.log(f"turntable_connect result ok={ok} msg={msg}")
        return ok, msg

    def home_turntable(self) -> Tuple[bool, str, Optional[float]]:
        debug_log.log("turntable_home start")
        res = turntable_service.home()
        try:
            self.turntable_message.emit(res.message)
        except Exception:
            pass
        debug_log.log(f"turntable_home done success={res.success} msg={res.message}")
        return res.success, res.message, res.offset_degrees

    def rotate_turntable(self, delta_deg: float) -> Tuple[bool, str]:
        try:
            debug_log.log(f"turntable_rotate start delta={delta_deg}")
            msg = turntable_service.move_relative(delta_deg)
            self.turntable_message.emit(msg)
            debug_log.log(f"turntable_rotate ok msg={msg}")
            return True, msg
        except Exception as ex:
            err = f"{ex}"
            try:
                self.turntable_message.emit(err)
            except Exception:
                pass
            debug_log.log(f"turntable_rotate failed err={err}")
            return False, err

    def set_turntable_step(self, step: float) -> None:
        try:
            st = state()
            st.turntable_step = float(step)
            save_state()
        except Exception:
            pass

    def _on_turntable_message(self, msg: str) -> None:
        try:
            self.turntable_message.emit(msg)
        except Exception:
            pass

    def _emit_plc_snapshot(self, snap) -> None:
        try:
            self.plc_snapshot.emit(snap)
        except Exception:
            pass

    # --- Axis ---------------------------------------------------------------
    def refresh_axis_devices(self) -> List[str]:
        try:
            return linear_axis_service.refresh_devices()
        except Exception:
            return []

    def connect_axis(self, port: str) -> Tuple[bool, str, Optional[int]]:
        debug_log.log(f"axis_connect start port={port}")
        ok = linear_axis_service.connect(port)
        endpoint = linear_axis_service.port_name() or port
        if ok:
            try:
                st = state()
                st.plc_host = str(port).split(":", 1)[0].strip() or str(port).strip()
                save_state()
            except Exception:
                pass
            try:
                hs = getattr(state(), "linear_axis_home_steps", None)
                if hs is None:
                    hs = getattr(state(), "linear_axis_last_steps", None)
            except Exception:
                hs = None
            self.axis_ready.emit(True)
            debug_log.log(f"axis_connect ok endpoint={endpoint}")
            return True, endpoint, hs
        debug_log.log(f"axis_connect failed endpoint={endpoint}")
        return False, endpoint, None

    def calibrate_axis(self, home_steps: int) -> None:
        def run():
            self.axis_calibrating.emit(True)
            try:
                debug_log.log(f"axis_calibrate start home_steps={home_steps}")
                res = linear_axis_service.calibrate_and_home(home_steps=home_steps)
                self.axis_log.emit(res.message)
                pos_steps = linear_axis_service.current_position_steps()
                total_steps = linear_axis_service.calibration_total_steps()
                self.axis_calibrated.emit(bool(res.success), pos_steps, total_steps)
                if res.success:
                    try:
                        st = state()
                        st.linear_axis_last_steps = int(pos_steps) if pos_steps is not None else None
                        save_state()
                    except Exception:
                        pass
            except Exception as ex:
                self.axis_log.emit(f"[Axis] Calibration failed: {ex}")
                debug_log.log(f"axis_calibrate failed err={ex}")
            finally:
                self.axis_calibrating.emit(False)
                debug_log.log("axis_calibrate done")

        threading.Thread(target=run, daemon=True).start()

    def home_axis(self, home_steps: int) -> None:
        def run():
            try:
                debug_log.log(f"axis_home start home_steps={home_steps}")
                res = linear_axis_service.home(home_steps=int(home_steps))
                self.axis_log.emit(res.message)
                if res.success:
                    pos_steps = linear_axis_service.current_position_steps()
                    total_steps = linear_axis_service.calibration_total_steps()
                    self.axis_calibrated.emit(True, pos_steps, total_steps)
                    try:
                        st = state()
                        st.linear_axis_last_steps = int(pos_steps) if pos_steps is not None else None
                        save_state()
                    except Exception:
                        pass
            except Exception as ex:
                self.axis_log.emit(f"[Axis] Home failed: {ex}")
                debug_log.log(f"axis_home failed err={ex}")

        threading.Thread(target=run, daemon=True).start()

    def goto_axis(self, target_steps: int) -> None:
        def run():
            try:
                debug_log.log(f"axis_goto start target_steps={target_steps}")
                res = linear_axis_service.goto_steps(int(target_steps))
                self.axis_log.emit(res.message)
                if res.success:
                    pos_steps = linear_axis_service.current_position_steps()
                    total_steps = linear_axis_service.calibration_total_steps()
                    self.axis_calibrated.emit(True, pos_steps, total_steps)
                    try:
                        st = state()
                        st.linear_axis_last_steps = int(pos_steps) if pos_steps is not None else None
                        save_state()
                    except Exception:
                        pass
            except Exception as ex:
                self.axis_log.emit(f"[Axis] Move failed: {ex}")
                debug_log.log(f"axis_goto failed err={ex}")

        threading.Thread(target=run, daemon=True).start()

    def set_axis_home(self, home_steps: int) -> None:
        try:
            st = state()
            st.linear_axis_home_steps = int(home_steps)
            save_state()
            self.axis_log.emit(f"[Axis] Home position set to {int(home_steps)} steps.")
            debug_log.log(f"axis_set_home steps={home_steps}")
        except Exception:
            pass

    def shutdown(self) -> None:
        try:
            if self._tt_listener:
                turntable_service.remove_listener(self._tt_listener)
        except Exception:
            pass
        try:
            plc_service.remove_status_listener(self._emit_plc_snapshot)
        except Exception:
            pass
        try:
            plc_service.disconnect()
        except Exception:
            pass
