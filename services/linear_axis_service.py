from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

from services import plc_service
from services.config import state


@dataclass
class LinearAxisResult:
    success: bool
    message: str
    position_steps: Optional[int] = None
    total_steps: Optional[int] = None

_act_lock = threading.Lock()


def refresh_devices() -> List[str]:
    """
    Backwards-compatible UI helper.
    For the PLC model, this returns a short list of known PLC hosts.
    """
    hosts: List[str] = []
    try:
        st = state()
        h = getattr(st, "plc_host", None)
        if h:
            hosts.append(str(h))
    except Exception:
        pass
    if not hosts:
        hosts.append("169.254.129.10")
    out: List[str] = []
    for h in hosts:
        h2 = str(h).strip()
        if h2 and h2 not in out:
            out.append(h2)
    return out


def connect(port: str, baud: int = 0) -> bool:  # baud kept for call compatibility
    host = str(port or "").strip()
    port_override: Optional[int] = None
    if ":" in host and host.count(":") == 1:
        h, p = host.split(":", 1)
        h = h.strip()
        try:
            port_override = int(p.strip())
        except Exception:
            port_override = None
        host = h
    try:
        st = state()
        plc_port = int(getattr(st, "plc_port", 502) or 502)
        unit_id = int(getattr(st, "plc_unit_id", 255) or 255)
    except Exception:
        plc_port = 502
        unit_id = 255
    if port_override is not None:
        plc_port = int(port_override)
    return plc_service.connect(host, port=plc_port, unit_id=unit_id, force_reconnect=True)


def disconnect() -> None:
    plc_service.disconnect()


def is_connected() -> bool:
    return plc_service.is_connected()


def port_name() -> Optional[str]:
    return plc_service.endpoint()


def add_listener(cb: Callable[[str], None]) -> None:
    # PLC emits messages via the shared plc_service channel.
    plc_service.add_message_listener(cb)


def remove_listener(cb: Callable[[str], None]) -> None:
    plc_service.remove_message_listener(cb)


def is_calibrated() -> bool:
    try:
        st = plc_service.last_status() or plc_service.read_status()
        return bool(getattr(st, "act_calib_valid", False))
    except Exception:
        return False


def calibration_total_steps() -> Optional[int]:
    try:
        st = plc_service.last_status() or plc_service.read_status()
        total = int(getattr(st, "act_calib_total_steps", 0) or 0)
        return total if total > 0 else None
    except Exception:
        return None


def current_position_steps() -> Optional[int]:
    try:
        st = plc_service.last_status() or plc_service.read_status()
        return int(getattr(st, "act_pos_steps", 0) or 0)
    except Exception:
        return None


def read_calibration_info() -> LinearAxisResult:
    if not plc_service.is_connected():
        return LinearAxisResult(False, "PLC not connected.", None, None)
    try:
        with _act_lock:
            plc_service.ensure_motion_ready()
            st = plc_service.read_status()
            cal = bool(getattr(st, "act_calib_valid", False))
            total = int(getattr(st, "act_calib_total_steps", 0) or 0)
            pos_steps = int(getattr(st, "act_pos_steps", 0) or 0)
            msg = f"[INFO] Calibrated: {'YES' if cal else 'NO'}"
            msg += f" [INFO] Current position: {pos_steps} steps"
            if total > 0:
                msg += f" [INFO] Total steps: {total}"
            return LinearAxisResult(True, msg, pos_steps, total if total > 0 else None)
    except Exception as ex:
        summ = ""
        try:
            summ = plc_service.status_summary(plc_service.last_status())
        except Exception:
            summ = ""
        extra = f" ({summ})" if summ else ""
        return LinearAxisResult(False, f"[INFO] Read failed: {ex}{extra}", None, None)


def calibrate_and_home(home_steps: Optional[int] = None, timeout_s: float = 180.0) -> LinearAxisResult:
    if not plc_service.is_connected():
        return LinearAxisResult(False, "PLC not connected.", current_position_steps(), calibration_total_steps())
    try:
        with _act_lock:
            plc_service.ensure_motion_ready()
            client = plc_service.client()
            st = plc_service.read_status()
            already_cal = bool(getattr(st, "act_calib_valid", False)) and int(getattr(st, "act_calib_total_steps", 0) or 0) > 0
            if not already_cal:
                client.actuator_calibrate(wait=True, timeout_s=float(timeout_s), poll_s=0.2)
                st = plc_service.read_status()
            total = int(getattr(st, "act_calib_total_steps", 0) or 0)
            if total <= 0:
                return LinearAxisResult(False, "[CAL] Calibration completed, but total steps is invalid.", None, None)

            target_steps = int(home_steps) if home_steps is not None else (total // 2)
            target_steps = max(0, min(int(target_steps), int(total)))
            client.actuator_goto(target_steps, wait=True, timeout_s=60.0, poll_s=0.2)
            st2 = plc_service.read_status()
            pos_steps = int(getattr(st2, "act_pos_steps", 0) or 0)
            if already_cal:
                msg = f"[CAL] Already calibrated; homed to {target_steps} steps."
            else:
                msg = f"[CAL] Calibration complete; homed to {target_steps} steps."
            return LinearAxisResult(True, msg, pos_steps, total)
    except Exception as ex:
        summ = ""
        try:
            summ = plc_service.status_summary(plc_service.last_status())
        except Exception:
            summ = ""
        extra = f" ({summ})" if summ else ""
        return LinearAxisResult(False, f"[CAL] Failed: {ex}{extra}", current_position_steps(), calibration_total_steps())


def goto_steps(target_steps: int, timeout_s: float = 60.0) -> LinearAxisResult:
    if not plc_service.is_connected():
        return LinearAxisResult(False, "PLC not connected.", current_position_steps(), calibration_total_steps())
    try:
        with _act_lock:
            plc_service.ensure_motion_ready()
            st = plc_service.read_status()
            if not bool(getattr(st, "act_calib_valid", False)):
                return LinearAxisResult(False, "[ERR] Axis not calibrated. Please run calibration first.", current_position_steps(), calibration_total_steps())
            total = int(getattr(st, "act_calib_total_steps", 0) or 0)
            if total <= 0:
                return LinearAxisResult(False, "[ERR] Invalid calibration (total steps is 0).", current_position_steps(), None)
            t = max(0, min(int(target_steps), int(total)))
            client = plc_service.client()
            client.actuator_goto(t, wait=True, timeout_s=float(timeout_s), poll_s=0.2)
            st2 = plc_service.read_status()
            pos_steps = int(getattr(st2, "act_pos_steps", 0) or 0)
            return LinearAxisResult(True, f"[MOVE] Position {pos_steps} steps.", pos_steps, total)
    except Exception as ex:
        summ = ""
        try:
            summ = plc_service.status_summary(plc_service.last_status())
        except Exception:
            summ = ""
        extra = f" ({summ})" if summ else ""
        return LinearAxisResult(False, f"[MOVE] Failed: {ex}{extra}", current_position_steps(), calibration_total_steps())


def home(home_steps: int, timeout_s: float = 60.0) -> LinearAxisResult:
    return goto_steps(home_steps, timeout_s=timeout_s)
