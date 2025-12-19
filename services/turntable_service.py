from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, List, Optional

from services import plc_service
from services.config import state


@dataclass
class TurntableHomeResult:
    success: bool
    message: str
    offset_degrees: Optional[float] = None

_tt_lock = threading.Lock()


def _parse_endpoint(text: str) -> tuple[str, Optional[int]]:
    # Accept either "host" or "host:port" in the UI field.
    s = str(text or "").strip()
    if not s:
        return "", None
    if ":" in s and s.count(":") == 1:
        host, port_s = s.split(":", 1)
        host = host.strip()
        try:
            port = int(port_s.strip())
        except Exception:
            port = None
        return host, port
    return s, None


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
    # Provide the typical default from the system guide if nothing is configured.
    if not hosts:
        hosts.append("169.254.129.10")
    # Deduplicate while preserving order.
    out: List[str] = []
    for h in hosts:
        h2 = str(h).strip()
        if h2 and h2 not in out:
            out.append(h2)
    return out


def connect(port: str, baud: int = 0) -> bool:  # baud kept for call compatibility
    host, port_override = _parse_endpoint(port)
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
    plc_service.add_message_listener(cb)


def remove_listener(cb: Callable[[str], None]) -> None:
    plc_service.remove_message_listener(cb)


def _require_connected():
    if not plc_service.is_connected():
        raise RuntimeError("PLC not connected.")


def home(timeout_s: float = 60.0) -> TurntableHomeResult:
    """
    UI 'Home' action for PLC turntable.

    The PLC does not perform a sensor-based home. We emulate the legacy UX by
    reading the current published turntable angle (wrapped to (-180, 180])
    and commanding a relative move by -current_position.
    """
    try:
        with _tt_lock:
            _require_connected()
            plc_service.ensure_motion_ready()
            st = plc_service.read_status()
            pos = float(getattr(st, "tt_pos_deg", 0.0) or 0.0)
            # Clamp tiny noise
            if abs(pos) < 1e-6:
                return TurntableHomeResult(True, "Homing complete (already at zero).", offset_degrees=0.0)
            delta = -pos
            client = plc_service.client()
            client.turntable_move_rel(delta, wait=True, timeout_s=float(timeout_s))
            # Re-read to report final position.
            st2 = plc_service.read_status()
            pos2 = float(getattr(st2, "tt_pos_deg", 0.0) or 0.0)
            return TurntableHomeResult(
                True,
                f"Homing complete. Rotated {delta:.3f} deg to zero (pos now {pos2:.3f}).",
                offset_degrees=pos,
            )
    except Exception as ex:
        summ = ""
        try:
            summ = plc_service.status_summary(plc_service.last_status())
        except Exception:
            summ = ""
        extra = f" ({summ})" if summ else ""
        return TurntableHomeResult(False, f"Homing failed: {ex}{extra}")


def move_relative(angle_deg: float, timeout_s: float = 60.0) -> str:
    try:
        with _tt_lock:
            _require_connected()
            plc_service.ensure_motion_ready()
            delta = float(angle_deg)
            if abs(delta) < 1e-6:
                return "Rotation skipped (below threshold)."
            client = plc_service.client()
            client.turntable_move_rel(delta, wait=True, timeout_s=float(timeout_s))
            return f"Rotated {delta:.2f} deg."
    except Exception as ex:
        summ = ""
        try:
            summ = plc_service.status_summary(plc_service.last_status())
        except Exception:
            summ = ""
        extra = f" ({summ})" if summ else ""
        raise RuntimeError(f"Turntable move failed: {ex}{extra}")
