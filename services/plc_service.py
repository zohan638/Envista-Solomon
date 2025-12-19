from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from services.config import state, save_state

try:
    from modbus_sdk.envista_modbus_client import (
        EnvistaClient,
        EnvistaStatus,
    )
except Exception:  # pragma: no cover
    EnvistaClient = None  # type: ignore[assignment]
    EnvistaStatus = None  # type: ignore[assignment]


StatusListener = Callable[["PlcSnapshot"], None]
MessageListener = Callable[[str], None]


@dataclass(frozen=True)
class PlcSnapshot:
    connected: bool
    host: Optional[str]
    port: Optional[int]
    unit_id: Optional[int]
    status: Optional["EnvistaStatus"]
    last_error: Optional[str]
    ts: float


class _PlcService:
    def __init__(self) -> None:
        self._client: Optional[EnvistaClient] = None
        self._host: Optional[str] = None
        self._port: int = 502
        self._unit_id: int = 255

        self._lock = threading.RLock()
        self._connected = False
        self._last_status: Optional[EnvistaStatus] = None
        self._last_error: Optional[str] = None

        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_period_s = 0.2

        self._msg_listeners: List[MessageListener] = []
        self._status_listeners: List[StatusListener] = []

    # --- listeners ---
    def add_message_listener(self, cb: MessageListener) -> None:
        self._msg_listeners.append(cb)

    def remove_message_listener(self, cb: MessageListener) -> None:
        try:
            self._msg_listeners.remove(cb)
        except ValueError:
            pass

    def add_status_listener(self, cb: StatusListener) -> None:
        self._status_listeners.append(cb)

    def remove_status_listener(self, cb: StatusListener) -> None:
        try:
            self._status_listeners.remove(cb)
        except ValueError:
            pass

    def _emit_msg(self, msg: str) -> None:
        if not msg:
            return
        for cb in list(self._msg_listeners):
            try:
                cb(msg)
            except Exception:
                pass

    def _emit_status(self) -> None:
        snap = PlcSnapshot(
            connected=self._connected,
            host=self._host,
            port=self._port if self._host else None,
            unit_id=self._unit_id if self._host else None,
            status=self._last_status,
            last_error=self._last_error,
            ts=time.time(),
        )
        for cb in list(self._status_listeners):
            try:
                cb(snap)
            except Exception:
                pass

    # --- connection ---
    def is_available(self) -> bool:
        return EnvistaClient is not None

    def is_connected(self) -> bool:
        return bool(self._connected and self._client is not None)

    def endpoint(self) -> Optional[str]:
        if not self._connected or not self._host:
            return None
        return f"{self._host}:{self._port} (unit {self._unit_id})"

    def last_status(self) -> Optional[EnvistaStatus]:
        return self._last_status

    def last_error(self) -> Optional[str]:
        return self._last_error

    def connect(
        self,
        host: str,
        *,
        port: int = 502,
        unit_id: int = 255,
        timeout_s: float = 1.5,
        poll_period_s: float = 0.2,
        heartbeat_period_s: float = 0.2,
        debug_enable: bool = True,
        allow_motion: bool = True,
        reset_faults: bool = True,
        force_reconnect: bool = False,
    ) -> bool:
        host = str(host or "").strip()
        if not host:
            self._last_error = "PLC host is empty."
            self._emit_status()
            return False

        if EnvistaClient is None:
            self._last_error = "Modbus client unavailable (missing pymodbus?)."
            self._emit_status()
            return False

        with self._lock:
            same_endpoint = (
                self._connected
                and self._client is not None
                and self._host == host
                and int(self._port) == int(port)
                and int(self._unit_id) == int(unit_id)
            )
            if same_endpoint and not force_reconnect:
                # Ensure poll thread is running and refresh status cache.
                self._poll_period_s = max(0.05, float(poll_period_s))
                try:
                    self._last_status = self._client.read_status()
                    self._last_error = None
                except Exception as ex:
                    self._last_error = f"Status read failed: {ex}"
                try:
                    self._client.start_heartbeat(period_s=float(heartbeat_period_s))
                except Exception:
                    pass
                if reset_faults:
                    try:
                        self._client.clear_halt()
                    except Exception:
                        pass
                    try:
                        self._client.reset_fault()
                    except Exception:
                        pass
                if debug_enable:
                    try:
                        self._client.set_debug_enable(True)
                    except Exception:
                        pass
                if allow_motion:
                    try:
                        self._client.set_allow_motion(True)
                    except Exception:
                        pass
                try:
                    self._last_status = self.ensure_motion_ready(
                        require_allow_motion=bool(allow_motion),
                        require_debug=False,
                        timeout_s=1.5,
                    )
                except Exception as ex:
                    self._emit_msg(f"[PLC] Connected, but not ready for motion: {ex}")
                self._start_poll_locked()
                self._emit_status()
                return True

            self._disconnect_locked(reason=None)

            self._host = host
            self._port = int(port)
            self._unit_id = int(unit_id)
            self._poll_period_s = max(0.05, float(poll_period_s))

            try:
                client = EnvistaClient(host, port=self._port, unit_id=self._unit_id, timeout_s=float(timeout_s))
                client.connect()
            except Exception as ex:
                self._client = None
                self._connected = False
                self._last_error = f"Connect failed: {ex}"
                self._emit_status()
                return False

            self._client = client
            self._connected = True
            self._last_error = None

            # Persist PLC endpoint for future sessions.
            try:
                st = state()
                st.plc_host = self._host
                st.plc_port = self._port
                st.plc_unit_id = self._unit_id
                save_state()
            except Exception:
                pass

            # Start watchdog heartbeat.
            try:
                client.start_heartbeat(period_s=float(heartbeat_period_s))
            except Exception:
                pass

            # Clear halt/fault so motion commands can be accepted after watchdog/door events.
            if reset_faults:
                try:
                    client.clear_halt()
                except Exception:
                    pass
                try:
                    client.reset_fault()
                except Exception:
                    pass

            # Enable debug so jog/motion controls are permitted (UI parity with legacy).
            if debug_enable:
                try:
                    client.set_debug_enable(True)
                except Exception:
                    pass

            # Enable motion gate by default so UI commands work.
            if allow_motion:
                try:
                    client.set_allow_motion(True)
                except Exception as ex:
                    self._emit_msg(f"[PLC] Connected, but ALLOW_MOTION failed: {ex}")

            # Prime status cache and start polling.
            try:
                self._last_status = self.ensure_motion_ready(
                    require_allow_motion=bool(allow_motion),
                    require_debug=False,
                    timeout_s=1.5,
                )
            except Exception as ex:
                self._emit_msg(f"[PLC] Connected, but not ready for motion: {ex}")
                try:
                    self._last_status = client.read_status()
                except Exception as ex2:
                    self._last_status = None
                    self._last_error = f"Initial status read failed: {ex2}"

            self._start_poll_locked()
            self._emit_msg(f"[PLC] Connected: {self.endpoint() or host}")
            self._emit_status()
            return True

    def disconnect(self) -> None:
        with self._lock:
            self._disconnect_locked(reason="[PLC] Disconnected.")

    def _disconnect_locked(self, *, reason: Optional[str]) -> None:
        self._stop_poll_locked()
        client = self._client
        self._client = None
        self._connected = False
        self._last_status = None
        if client is not None:
            try:
                client.stop_heartbeat()
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass
        if reason:
            self._emit_msg(reason)
        self._emit_status()

    def _start_poll_locked(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._poll_stop.clear()

        def _run():
            while not self._poll_stop.is_set():
                time.sleep(self._poll_period_s)
                try:
                    self._poll_once()
                except Exception:
                    # _poll_once is already defensive; never crash the UI.
                    pass

        self._poll_thread = threading.Thread(target=_run, name="EnvistaPlcPoll", daemon=True)
        self._poll_thread.start()

    def _stop_poll_locked(self) -> None:
        self._poll_stop.set()
        if self._poll_thread and self._poll_thread.is_alive():
            # Avoid deadlock if the poll thread detects its own failure.
            if threading.current_thread() is not self._poll_thread:
                self._poll_thread.join(timeout=1.0)
        self._poll_thread = None

    def _poll_once(self) -> None:
        client = self._client
        if client is None or not self._connected:
            return
        try:
            st = client.read_status()
            self._last_status = st
            self._last_error = None
            self._emit_status()
        except Exception as ex:
            # Treat as disconnect; allow user-driven reconnect.
            with self._lock:
                self._last_error = f"Status poll failed: {ex}"
                self._emit_msg(f"[PLC] Connection lost: {ex}")
                self._disconnect_locked(reason=None)

    # --- safe access ---
    def require_client(self) -> EnvistaClient:
        if self._client is None or not self._connected:
            raise RuntimeError("PLC not connected.")
        return self._client

    def read_status(self) -> EnvistaStatus:
        client = self.require_client()
        st = client.read_status()
        self._last_status = st
        return st

    # --- system commands (thin wrappers) ---
    def set_allow_motion(self, enabled: bool) -> None:
        client = self.require_client()
        client.set_allow_motion(bool(enabled))

    def reset_fault(self) -> None:
        client = self.require_client()
        client.reset_fault()

    def halt_all(self) -> None:
        client = self.require_client()
        client.halt_all()

    def clear_halt(self) -> None:
        client = self.require_client()
        client.clear_halt()

    def ensure_motion_ready(self, *, require_allow_motion: bool = True, require_debug: bool = False, timeout_s: float = 2.0) -> "EnvistaStatus":
        """
        Bring the PLC into a state where motion commands are expected to be accepted.

        - Best-effort clears HALT/FAULT.
        - Blocks on door interlock.
        - Ensures watchdog is OK (heartbeat running).
        - Ensures DEBUG_ENABLE and ALLOW_MOTION gates are asserted (optional).
        """
        client = self.require_client()
        deadline = time.time() + float(timeout_s)

        st = client.read_status()
        if bool(getattr(st, "door_open", False)):
            raise RuntimeError("Door open (interlock active). Close door and Reset Fault.")

        # Ensure watchdog/heartbeat is OK. This may be false right after connect if the PLC
        # has not yet observed the heartbeat change.
        if hasattr(st, "watchdog_ok") and not bool(getattr(st, "watchdog_ok", True)):
            try:
                client.start_heartbeat()
            except Exception:
                pass
            while time.time() < deadline:
                time.sleep(0.05)
                st = client.read_status()
                if bool(getattr(st, "watchdog_ok", False)):
                    break
            if not bool(getattr(st, "watchdog_ok", True)):
                raise RuntimeError("Watchdog not OK (heartbeat missing).")

        # Try to clear common non-hardware stops (watchdog must be OK for this to stick).
        try:
            if bool(getattr(st, "sys_halted", False)):
                client.clear_halt()
        except Exception:
            pass
        try:
            if bool(getattr(st, "sys_fault", False)):
                client.reset_fault()
        except Exception:
            pass

        st = client.read_status()
        if bool(getattr(st, "door_open", False)):
            raise RuntimeError("Door open (interlock active). Close door and Reset Fault.")
        if bool(getattr(st, "sys_halted", False)):
            raise RuntimeError("System is HALTED. Clear halt required.")
        if bool(getattr(st, "sys_fault", False)):
            code = int(getattr(st, "sys_fault_code", 0) or 0)
            raise RuntimeError(f"System FAULT (code {code}). Reset fault required.")

        # Ensure gate bits (some faults can clear them).
        last = st
        while time.time() < deadline:
            need_write = False
            try:
                if require_debug and not bool(getattr(last, "debug_enabled", False)):
                    need_write = True
                    client.set_debug_enable(True)
            except Exception:
                pass
            try:
                if require_allow_motion and not bool(getattr(last, "allow_motion_active", False)):
                    need_write = True
                    client.set_allow_motion(True)
            except Exception:
                pass

            if not need_write:
                return last

            time.sleep(0.05)
            last = client.read_status()

        # One last read to improve error reporting.
        last = client.read_status()
        dbg_ok = bool(getattr(last, "debug_enabled", True))
        allow_ok = bool(getattr(last, "allow_motion_active", True))
        raise RuntimeError(
            "PLC motion gates not active "
            f"(DEBUG={dbg_ok}, ALLOW_MOTION={allow_ok})."
        )

    # --- waits ---
    def wait_for_ack(
        self,
        *,
        cmd_seq_getter: Callable[[EnvistaStatus], int],
        ack_seq_getter: Callable[[EnvistaStatus], int],
        expect_seq: int,
        timeout_s: float,
        poll_s: float = 0.05,
    ) -> EnvistaStatus:
        deadline = time.time() + float(timeout_s)
        last = self.read_status()
        while time.time() < deadline:
            if ack_seq_getter(last) == (expect_seq & 0xFFFF):
                return last
            time.sleep(float(poll_s))
            last = self.read_status()
        raise TimeoutError(f"PLC ack timeout (expect {expect_seq}, got {ack_seq_getter(last)})")


_plc = _PlcService()


# Public API
def is_available() -> bool:
    return _plc.is_available()


def connect(host: str, *, port: int = 502, unit_id: int = 255, force_reconnect: bool = False) -> bool:
    return _plc.connect(host, port=port, unit_id=unit_id, force_reconnect=bool(force_reconnect))


def disconnect() -> None:
    _plc.disconnect()


def is_connected() -> bool:
    return _plc.is_connected()


def endpoint() -> Optional[str]:
    return _plc.endpoint()


def last_status() -> Optional["EnvistaStatus"]:
    return _plc.last_status()


def last_error() -> Optional[str]:
    return _plc.last_error()


def add_message_listener(cb: MessageListener) -> None:
    _plc.add_message_listener(cb)


def remove_message_listener(cb: MessageListener) -> None:
    _plc.remove_message_listener(cb)


def add_status_listener(cb: StatusListener) -> None:
    _plc.add_status_listener(cb)


def remove_status_listener(cb: StatusListener) -> None:
    _plc.remove_status_listener(cb)


def read_status() -> "EnvistaStatus":
    return _plc.read_status()


def client() -> "EnvistaClient":
    return _plc.require_client()


def set_allow_motion(enabled: bool) -> None:
    _plc.set_allow_motion(enabled)


def reset_fault() -> None:
    _plc.reset_fault()


def halt_all() -> None:
    _plc.halt_all()


def clear_halt() -> None:
    _plc.clear_halt()


def ensure_motion_ready(*, require_allow_motion: bool = True, require_debug: bool = False, timeout_s: float = 2.0) -> "EnvistaStatus":
    return _plc.ensure_motion_ready(require_allow_motion=require_allow_motion, require_debug=require_debug, timeout_s=float(timeout_s))


def status_summary(status: Optional["EnvistaStatus"]) -> str:
    """
    Compact, user-facing PLC snapshot string for logs/errors.
    Safe to call even when status is None.
    """
    if status is None:
        return ""
    try:
        sys_state = int(getattr(status, "sys_state", 0) or 0)
        ready = bool(getattr(status, "sys_ready", False))
        running = bool(getattr(status, "sys_running", False))
        fault = bool(getattr(status, "sys_fault", False))
        fault_code = int(getattr(status, "sys_fault_code", 0) or 0)
        halted = bool(getattr(status, "sys_halted", False))
        door_open = bool(getattr(status, "door_open", False))
        watchdog_ok = bool(getattr(status, "watchdog_ok", True))
        debug = bool(getattr(status, "debug_enabled", False))
        allow = bool(getattr(status, "allow_motion_active", False))

        act_pos = int(getattr(status, "act_pos_steps", 0) or 0)
        act_cal = bool(getattr(status, "act_calib_valid", False))
        act_total = int(getattr(status, "act_calib_total_steps", 0) or 0)
        act_moving = bool(getattr(status, "act_in_motion", False))

        tt_pos = float(getattr(status, "tt_pos_deg", 0.0) or 0.0)
        tt_moving = bool(getattr(status, "tt_in_motion", False))

        return (
            f"SYS(state={sys_state}, ready={ready}, running={running}, fault={fault}:{fault_code}, halted={halted}, door_open={door_open}, watchdog_ok={watchdog_ok}, debug={debug}, allow_motion={allow}) "
            f"ACT(pos_steps={act_pos}, calib={act_cal}, total_steps={act_total}, moving={act_moving}) "
            f"TT(pos_deg={tt_pos:.3f}, moving={tt_moving})"
        )
    except Exception:
        return ""
