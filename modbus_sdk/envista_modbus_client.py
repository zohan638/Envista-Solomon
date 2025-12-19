"""envista_modbus_client.py

Small, dependency-light Python helper for controlling the **Envista Opta** PLC
over **Modbus TCP** using `pymodbus`.

It implements:
 - Register/bit constants matching the finalized PLC firmware
 - Little-endian DINT packing/unpacking (low word first)
 - CmdSeq/AckSeq handshake helpers
 - A background master-heartbeat thread (watchdog keepalive)

Typical usage
-------------

```python
from envista_modbus_client import EnvistaClient

with EnvistaClient("169.254.129.10", unit_id=255) as plc:
    plc.start_heartbeat(period_s=0.2)
    plc.set_allow_motion(True)

    plc.actuator_calibrate(wait=True)
    st = plc.read_status()
    plc.actuator_goto(st.act_calib_total_steps // 2, wait=True)

    plc.turntable_move_rel(15.0)
```

Notes
-----
 - The PLC uses a watchdog: you must continuously update `svSysMasterHeartbeat`.
 - Addresses below are the same as the debug UI defaults:
     * status base = 24575 (0x6000 - 1)
     * param  base = 16383 (0x4000 - 1)
   Some Modbus tools use 1-based addressing; adjust if needed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import List, Optional

try:
    # pymodbus >= 3
    from pymodbus.client import ModbusTcpClient
except Exception:  # pragma: no cover
    from pymodbus.client.sync import ModbusTcpClient  # type: ignore


# ---------------------------------------------------------------------------
# Register map (Status Variables)
# ---------------------------------------------------------------------------
SV_BLOCK_LEN = 33

O_SYS_CMD_WORD = 0
O_SYS_CMD_SEQ = 1
O_SYS_ACK_SEQ = 2
O_SYS_STATUS_WORD = 3
O_SYS_STATE = 4
O_SYS_FAULT_CODE = 5
O_SYS_PLC_HB = 6
O_SYS_MASTER_HB = 7

O_ACT_CMD_WORD = 8
O_ACT_CMD_SEQ = 9
O_ACT_ACK_SEQ = 10
O_ACT_TARGET_DINT = 11  # (11..12)
O_ACT_POS_DINT = 13     # (13..14)
O_ACT_STATUS_WORD = 15
O_ACT_STATE = 16
O_ACT_FAULT_CODE = 17
O_ACT_IN_MOTION = 18
O_ACT_CAL_VALID = 19
O_ACT_CAL_TOTAL_DINT = 20  # (20..21)
O_ACT_CAL_SIG = 22

O_TT_CMD_WORD = 23
O_TT_CMD_SEQ = 24
O_TT_ACK_SEQ = 25
O_TT_TARGET_DINT = 26  # (26..27)
O_TT_POS_DINT = 28     # (28..29)
O_TT_STATUS_WORD = 30
O_TT_STATE = 31
O_TT_FAULT_CODE = 32


# ---------------------------------------------------------------------------
# Register map (Parameters)
# ---------------------------------------------------------------------------
PAR_BLOCK_LEN = 23  # offsets 0..22

P_MASTER_WATCHDOG_MS = 3

P_ACT_CALIB_STEPFREQ = 7
P_ACT_CALIB_ACCEL = 8
P_ACT_MAX_STEPFREQ = 9
P_ACT_ACCEL = 10
P_ACT_DECEL = 11
P_ACT_STOP_DECEL = 12

P_TT_MAX_STEPFREQ = 14
P_TT_ACCEL = 15
P_TT_DECEL = 16
P_TT_STOP_DECEL = 17
P_TT_DIR_POLARITY = 18

P_STEP_DUTY_PERMIL = 19
P_DIR_SETUP_DELAY_MS = 20
P_PWM_MIN_PERIOD_US = 21
P_PWM_OFF_PERIOD_US = 22


# ---------------------------------------------------------------------------
# Command bits
# ---------------------------------------------------------------------------
SYS_CMD_RESET_FAULT = 0x0001
SYS_CMD_HALT_ALL = 0x0002
SYS_CMD_CLEAR_HALT = 0x0004
SYS_CMD_ALLOW_MOTION = 0x0010
SYS_CMD_DEBUG_ENABLE = 0x0020

ACT_CMD_CALIBRATE = 0x0001
ACT_CMD_GOTO_ABS = 0x0002
ACT_CMD_HALT = 0x0004
ACT_CMD_JOG_POS = 0x0010
ACT_CMD_JOG_NEG = 0x0020

TT_CMD_MOVE_REL = 0x0001
TT_CMD_HALT = 0x0002
TT_CMD_JOG_CW = 0x0004
TT_CMD_JOG_CCW = 0x0008
TT_CMD_SET_HOME = 0x0010
TT_CMD_RESET_HOME = 0x0020


# ---------------------------------------------------------------------------
# Helpers (Opta uses low word first for DINT)
# ---------------------------------------------------------------------------
def unpack_dint_le(words: List[int], offset: int) -> int:
    """Unpack a signed 32-bit integer from two 16-bit registers (low word first)."""
    if offset + 1 >= len(words):
        return 0
    lo = int(words[offset]) & 0xFFFF
    hi = int(words[offset + 1]) & 0xFFFF
    u32 = (hi << 16) | lo
    if u32 & 0x80000000:
        return -((~u32 & 0xFFFFFFFF) + 1)
    return int(u32)


def pack_dint_le(value: int) -> List[int]:
    """Pack a signed 32-bit integer into two 16-bit registers (low word first)."""
    v = int(value) & 0xFFFFFFFF
    return [v & 0xFFFF, (v >> 16) & 0xFFFF]


def normalize_angle_deg(deg: float) -> float:
    """Normalize into (-180, 180] degrees."""
    x = float(deg)
    while x <= -180.0:
        x += 360.0
    while x > 180.0:
        x -= 360.0
    # avoid negative zero
    if abs(x) < 1e-12:
        x = 0.0
    return x


# ---------------------------------------------------------------------------
# Dataclasses (decoded views)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnvistaStatus:
    regs: List[int]
    ts: float

    # system
    @property
    def sys_cmd_seq(self) -> int:
        return int(self.regs[O_SYS_CMD_SEQ]) & 0xFFFF

    @property
    def sys_ack_seq(self) -> int:
        return int(self.regs[O_SYS_ACK_SEQ]) & 0xFFFF

    @property
    def sys_cmd_word(self) -> int:
        return int(self.regs[O_SYS_CMD_WORD]) & 0xFFFF

    @property
    def sys_status_word(self) -> int:
        return int(self.regs[O_SYS_STATUS_WORD]) & 0xFFFF

    @property
    def sys_state(self) -> int:
        return int(self.regs[O_SYS_STATE]) & 0xFFFF

    @property
    def sys_fault_code(self) -> int:
        return int(self.regs[O_SYS_FAULT_CODE]) & 0xFFFF

    @property
    def plc_heartbeat(self) -> int:
        return int(self.regs[O_SYS_PLC_HB]) & 0xFFFF

    @property
    def sys_ready(self) -> bool:
        return bool(self.sys_status_word & 0x0001)

    @property
    def sys_running(self) -> bool:
        return bool(self.sys_status_word & 0x0002)

    @property
    def sys_fault(self) -> bool:
        return bool(self.sys_status_word & 0x0004)

    @property
    def sys_halted(self) -> bool:
        return bool(self.sys_status_word & 0x0008)

    @property
    def watchdog_ok(self) -> bool:
        return bool(self.sys_status_word & 0x0010)

    @property
    def debug_enabled(self) -> bool:
        return bool(self.sys_status_word & 0x0020)

    @property
    def allow_motion_active(self) -> bool:
        return bool(self.sys_status_word & 0x0040)

    @property
    def door_open(self) -> bool:
        return bool(self.sys_status_word & 0x0080)

    @property
    def door_latched(self) -> bool:
        return bool(self.sys_status_word & 0x0100)

    # actuator
    @property
    def act_cmd_seq(self) -> int:
        return int(self.regs[O_ACT_CMD_SEQ]) & 0xFFFF

    @property
    def act_ack_seq(self) -> int:
        return int(self.regs[O_ACT_ACK_SEQ]) & 0xFFFF

    @property
    def act_pos_steps(self) -> int:
        return unpack_dint_le(self.regs, O_ACT_POS_DINT)

    @property
    def act_target_steps(self) -> int:
        return unpack_dint_le(self.regs, O_ACT_TARGET_DINT)

    @property
    def act_calib_total_steps(self) -> int:
        return unpack_dint_le(self.regs, O_ACT_CAL_TOTAL_DINT)

    @property
    def act_calib_valid(self) -> bool:
        return (int(self.regs[O_ACT_CAL_VALID]) & 0xFFFF) != 0

    @property
    def act_in_motion(self) -> bool:
        return (int(self.regs[O_ACT_IN_MOTION]) & 0xFFFF) != 0

    @property
    def act_state(self) -> int:
        return int(self.regs[O_ACT_STATE]) & 0xFFFF

    @property
    def act_fault_code(self) -> int:
        return int(self.regs[O_ACT_FAULT_CODE]) & 0xFFFF

    # turntable
    @property
    def tt_cmd_seq(self) -> int:
        return int(self.regs[O_TT_CMD_SEQ]) & 0xFFFF

    @property
    def tt_ack_seq(self) -> int:
        return int(self.regs[O_TT_ACK_SEQ]) & 0xFFFF

    @property
    def tt_pos_deg(self) -> float:
        return unpack_dint_le(self.regs, O_TT_POS_DINT) / 1000.0

    @property
    def tt_in_motion(self) -> bool:
        return bool(int(self.regs[O_TT_STATUS_WORD]) & 0x0001)

    @property
    def tt_home_reset_mode(self) -> bool:
        return bool(int(self.regs[O_TT_STATUS_WORD]) & 0x0002)

    @property
    def tt_state(self) -> int:
        return int(self.regs[O_TT_STATE]) & 0xFFFF

    @property
    def tt_fault_code(self) -> int:
        return int(self.regs[O_TT_FAULT_CODE]) & 0xFFFF


@dataclass(frozen=True)
class EnvistaParameters:
    regs: List[int]
    ts: float

    def u16(self, off: int) -> int:
        return int(self.regs[off]) & 0xFFFF

    def b(self, off: int) -> bool:
        return self.u16(off) != 0

    # common
    @property
    def master_watchdog_ms(self) -> int:
        return self.u16(P_MASTER_WATCHDOG_MS)

    # actuator
    @property
    def act_calib_stepfreq(self) -> int:
        return self.u16(P_ACT_CALIB_STEPFREQ)

    @property
    def act_calib_accel(self) -> int:
        return self.u16(P_ACT_CALIB_ACCEL)

    @property
    def act_max_stepfreq(self) -> int:
        return self.u16(P_ACT_MAX_STEPFREQ)

    @property
    def act_accel(self) -> int:
        return self.u16(P_ACT_ACCEL)

    @property
    def act_decel(self) -> int:
        return self.u16(P_ACT_DECEL)

    @property
    def act_stop_decel(self) -> int:
        return self.u16(P_ACT_STOP_DECEL)

    # turntable
    @property
    def tt_max_stepfreq(self) -> int:
        return self.u16(P_TT_MAX_STEPFREQ)

    @property
    def tt_accel(self) -> int:
        return self.u16(P_TT_ACCEL)

    @property
    def tt_decel(self) -> int:
        return self.u16(P_TT_DECEL)

    @property
    def tt_stop_decel(self) -> int:
        return self.u16(P_TT_STOP_DECEL)

    @property
    def tt_dir_polarity(self) -> bool:
        return self.b(P_TT_DIR_POLARITY)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class EnvistaClient:
    """High-level Modbus TCP client for the Envista PLC."""

    def __init__(
        self,
        host: str,
        port: int = 502,
        unit_id: int = 255,
        sv_base: int = 24575,
        par_base: int = 16383,
        timeout_s: float = 1.5,
    ):
        self.host = host
        self.port = int(port)
        self.unit_id = int(unit_id)
        self.sv_base = int(sv_base)
        self.par_base = int(par_base)
        self.timeout_s = float(timeout_s)

        self._client: Optional[ModbusTcpClient] = None
        self._last_status: Optional[EnvistaStatus] = None
        self._lock = threading.RLock()

        self._hb_thread: Optional[threading.Thread] = None
        self._hb_stop = threading.Event()
        self._hb_counter = 0

    # --- context manager ---
    def __enter__(self) -> "EnvistaClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop_heartbeat()
        self.close()

    # --- pymodbus call compat ---
    def _call(self, fn_name: str, *args, **kwargs):
        with self._lock:
            if self._client is None:
                raise RuntimeError("Not connected")
            fn = getattr(self._client, fn_name)

            for key in ("device_id", "unit", "slave"):
                try:
                    kw = dict(kwargs)
                    kw[key] = self.unit_id
                    return fn(*args, **kw)
                except TypeError:
                    continue
            return fn(*args, **kwargs)

    # --- connection ---
    def connect(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            self._client = ModbusTcpClient(self.host, port=self.port, timeout=self.timeout_s)
            if not self._client.connect():
                self._client = None
                raise ConnectionError(f"Failed to connect to {self.host}:{self.port}")

    def close(self) -> None:
        with self._lock:
            if self._client is None:
                return
            try:
                self._client.close()
            finally:
                self._client = None

    # --- low-level read/write ---
    def read_holding(self, address: int, count: int) -> List[int]:
        with self._lock:
            rr = self._call("read_holding_registers", int(address), count=int(count))
            if rr is None or getattr(rr, "isError", lambda: True)():
                raise IOError(f"read_holding_registers failed at {address} count={count}: {rr}")
            return list(rr.registers)  # type: ignore[attr-defined]

    def write_reg(self, address: int, value: int) -> None:
        with self._lock:
            rr = self._call("write_register", int(address), int(value) & 0xFFFF)
            if rr is None or getattr(rr, "isError", lambda: True)():
                raise IOError(f"write_register failed at {address}: {rr}")

    def write_regs(self, address: int, values: List[int]) -> None:
        with self._lock:
            rr = self._call("write_registers", int(address), [int(v) & 0xFFFF for v in values])
            if rr is None or getattr(rr, "isError", lambda: True)():
                raise IOError(f"write_registers failed at {address}: {rr}")

    # --- heartbeat ---
    def start_heartbeat(self, period_s: float = 0.2) -> None:
        """Start a background thread that increments the master heartbeat."""
        if self._hb_thread and self._hb_thread.is_alive():
            return

        # Write once synchronously so the PLC can immediately see the heartbeat change.
        try:
            self._hb_counter = (self._hb_counter + 1) & 0xFFFF
            self.write_reg(self.sv_base + O_SYS_MASTER_HB, self._hb_counter)
        except Exception:
            # Heartbeat thread will continue attempting writes.
            pass

        self._hb_stop.clear()

        def _run():
            next_t = time.time()
            while not self._hb_stop.is_set():
                now = time.time()
                if now >= next_t:
                    next_t = now + float(period_s)
                    try:
                        self._hb_counter = (self._hb_counter + 1) & 0xFFFF
                        self.write_reg(self.sv_base + O_SYS_MASTER_HB, self._hb_counter)
                    except Exception:
                        # Don't crash the app; let caller detect faults via status polling.
                        pass
                time.sleep(0.01)

        self._hb_thread = threading.Thread(target=_run, name="EnvistaHeartbeat", daemon=True)
        self._hb_thread.start()

    def stop_heartbeat(self) -> None:
        self._hb_stop.set()
        if self._hb_thread and self._hb_thread.is_alive():
            self._hb_thread.join(timeout=1.0)
        self._hb_thread = None

    # --- status/params ---
    def read_status(self) -> EnvistaStatus:
        regs = self.read_holding(self.sv_base, SV_BLOCK_LEN)
        st = EnvistaStatus(regs=regs, ts=time.time())
        self._last_status = st
        return st

    def read_parameters(self) -> EnvistaParameters:
        regs = self.read_holding(self.par_base, PAR_BLOCK_LEN)
        return EnvistaParameters(regs=regs, ts=time.time())

    def write_parameter_u16(self, offset: int, value: int) -> None:
        self.write_reg(self.par_base + int(offset), int(value) & 0xFFFF)

    def write_parameter_bool(self, offset: int, value: bool) -> None:
        self.write_reg(self.par_base + int(offset), 1 if value else 0)

    # --- handshake helpers ---
    def _next_seq(self, seq_offset: int) -> int:
        # Prefer cached status (fast), otherwise do a single-register read.
        if self._last_status is not None:
            curr = int(self._last_status.regs[seq_offset]) & 0xFFFF
        else:
            curr = int(self.read_holding(self.sv_base + seq_offset, 1)[0]) & 0xFFFF
        return (curr + 1) & 0xFFFF

    def _wait_ack(self, ack_offset: int, expect_seq: int, *, timeout_s: float = 2.0, poll_s: float = 0.05) -> EnvistaStatus:
        """Wait until the PLC ack counter matches the expected seq."""
        exp = int(expect_seq) & 0xFFFF
        deadline = time.time() + float(timeout_s)
        last = self.read_status()
        while time.time() < deadline:
            got = int(last.regs[ack_offset]) & 0xFFFF
            if got == exp:
                return last
            time.sleep(float(poll_s))
            last = self.read_status()
        raise TimeoutError(f"Ack timeout (off={ack_offset}, expect={exp}, got={int(last.regs[ack_offset]) & 0xFFFF})")

    # --- system commands ---
    def _send_sys_cmd(self, action_bits: int = 0, *, allow_motion: Optional[bool] = None, debug_enable: Optional[bool] = None) -> None:
        st = self.read_status()  # keeps cache fresh
        cmd = st.sys_cmd_word

        # Update sticky flag bits if requested
        if allow_motion is not None:
            if allow_motion:
                cmd |= SYS_CMD_ALLOW_MOTION
            else:
                cmd &= ~SYS_CMD_ALLOW_MOTION
        if debug_enable is not None:
            if debug_enable:
                cmd |= SYS_CMD_DEBUG_ENABLE
            else:
                cmd &= ~SYS_CMD_DEBUG_ENABLE

        # Apply action bits (edge-triggered via seq)
        cmd = (cmd & (SYS_CMD_ALLOW_MOTION | SYS_CMD_DEBUG_ENABLE)) | (action_bits & 0xFFFF)
        seq = self._next_seq(O_SYS_CMD_SEQ)
        self.write_reg(self.sv_base + O_SYS_CMD_WORD, cmd)
        self.write_reg(self.sv_base + O_SYS_CMD_SEQ, seq)
        try:
            self._wait_ack(O_SYS_ACK_SEQ, seq, timeout_s=1.0)
        except Exception:
            # Status polling can surface faults; avoid hard-failing UI toggles.
            pass

    def set_allow_motion(self, enabled: bool) -> None:
        self._send_sys_cmd(0, allow_motion=bool(enabled))

    def set_debug_enable(self, enabled: bool) -> None:
        self._send_sys_cmd(0, debug_enable=bool(enabled))

    def reset_fault(self) -> None:
        self._send_sys_cmd(SYS_CMD_RESET_FAULT)

    def halt_all(self) -> None:
        self._send_sys_cmd(SYS_CMD_HALT_ALL)

    def clear_halt(self) -> None:
        self._send_sys_cmd(SYS_CMD_CLEAR_HALT)

    # --- actuator commands ---
    def actuator_calibrate(self, *, wait: bool = False, timeout_s: float = 180.0, poll_s: float = 0.2) -> None:
        pre = self.read_status()
        pre_cal_valid = bool(pre.act_calib_valid)

        seq = self._next_seq(O_ACT_CMD_SEQ)
        self.write_reg(self.sv_base + O_ACT_CMD_WORD, ACT_CMD_CALIBRATE)
        self.write_reg(self.sv_base + O_ACT_CMD_SEQ, seq)
        self._wait_ack(O_ACT_ACK_SEQ, seq, timeout_s=2.0)
        if wait:
            deadline = time.time() + float(timeout_s)
            start_deadline = min(deadline, time.time() + 2.0)
            saw_motion = False
            last = self.read_status()
            while time.time() < deadline:
                last = self.read_status()
                if last.act_in_motion:
                    saw_motion = True

                if saw_motion:
                    if last.act_calib_valid and not last.act_in_motion:
                        return
                else:
                    # If already calibrated and nothing starts, treat as no-op.
                    if pre_cal_valid and last.act_calib_valid and not last.act_in_motion and time.time() >= start_deadline:
                        return
                    # If not calibrated, calibration must start promptly.
                    if (not pre_cal_valid) and time.time() >= start_deadline and (not last.act_in_motion):
                        raise TimeoutError("Calibration did not start (axis never entered motion).")

                time.sleep(float(poll_s))

            raise TimeoutError("Calibration did not complete before timeout.")

    def actuator_halt(self) -> None:
        seq = self._next_seq(O_ACT_CMD_SEQ)
        self.write_reg(self.sv_base + O_ACT_CMD_WORD, ACT_CMD_HALT)
        self.write_reg(self.sv_base + O_ACT_CMD_SEQ, seq)
        self._wait_ack(O_ACT_ACK_SEQ, seq, timeout_s=2.0)

    def actuator_goto(self, target_steps: int, *, wait: bool = False, timeout_s: float = 60.0, poll_s: float = 0.2) -> None:
        # Clamp to known range if calibrated
        st = self.read_status()
        t = int(target_steps)
        if st.act_calib_total_steps > 0:
            t = max(0, min(t, st.act_calib_total_steps))

        start_pos = int(st.act_pos_steps)

        self.write_regs(self.sv_base + O_ACT_TARGET_DINT, pack_dint_le(t))
        seq = self._next_seq(O_ACT_CMD_SEQ)
        self.write_reg(self.sv_base + O_ACT_CMD_WORD, ACT_CMD_GOTO_ABS)
        self.write_reg(self.sv_base + O_ACT_CMD_SEQ, seq)
        self._wait_ack(O_ACT_ACK_SEQ, seq, timeout_s=2.0)

        if wait:
            deadline = time.time() + float(timeout_s)
            start_deadline = min(deadline, time.time() + 1.5)
            saw_motion = False
            tol_steps = 2

            last = self.read_status()
            while time.time() < deadline:
                last = self.read_status()
                pos = int(last.act_pos_steps)
                moving = bool(last.act_in_motion)

                if moving or abs(pos - start_pos) >= tol_steps:
                    saw_motion = True

                # Done condition: stopped and at target (or very near).
                if (not moving) and abs(pos - t) <= tol_steps:
                    return

                # If we never observe motion and we aren't at target shortly after command, assume rejected.
                if (not saw_motion) and (time.time() >= start_deadline) and (not moving) and abs(pos - t) > tol_steps:
                    raise TimeoutError("Actuator did not start moving (no motion observed).")

                time.sleep(float(poll_s))

            raise TimeoutError("Actuator move did not complete before timeout.")

    def actuator_jog_pos(self) -> None:
        seq = self._next_seq(O_ACT_CMD_SEQ)
        self.write_reg(self.sv_base + O_ACT_CMD_WORD, ACT_CMD_JOG_POS)
        self.write_reg(self.sv_base + O_ACT_CMD_SEQ, seq)
        self._wait_ack(O_ACT_ACK_SEQ, seq, timeout_s=2.0)

    def actuator_jog_neg(self) -> None:
        seq = self._next_seq(O_ACT_CMD_SEQ)
        self.write_reg(self.sv_base + O_ACT_CMD_WORD, ACT_CMD_JOG_NEG)
        self.write_reg(self.sv_base + O_ACT_CMD_SEQ, seq)
        self._wait_ack(O_ACT_ACK_SEQ, seq, timeout_s=2.0)

    # --- turntable commands ---
    def turntable_halt(self) -> None:
        seq = self._next_seq(O_TT_CMD_SEQ)
        self.write_reg(self.sv_base + O_TT_CMD_WORD, TT_CMD_HALT)
        self.write_reg(self.sv_base + O_TT_CMD_SEQ, seq)
        self._wait_ack(O_TT_ACK_SEQ, seq, timeout_s=2.0)

    def turntable_move_rel(self, delta_deg: float, *, wait: bool = False, timeout_s: float = 60.0, poll_s: float = 0.2) -> None:
        start = self.read_status()
        start_pos = float(start.tt_pos_deg)
        expected = normalize_angle_deg(start_pos + float(delta_deg))

        deg_x1000 = int(round(float(delta_deg) * 1000.0))
        self.write_regs(self.sv_base + O_TT_TARGET_DINT, pack_dint_le(deg_x1000))
        seq = self._next_seq(O_TT_CMD_SEQ)
        self.write_reg(self.sv_base + O_TT_CMD_WORD, TT_CMD_MOVE_REL)
        self.write_reg(self.sv_base + O_TT_CMD_SEQ, seq)
        self._wait_ack(O_TT_ACK_SEQ, seq, timeout_s=2.0)
        if wait:
            deadline = time.time() + float(timeout_s)
            start_deadline = min(deadline, time.time() + 1.5)
            saw_motion = False
            tol_deg = 0.25

            last = self.read_status()
            while time.time() < deadline:
                last = self.read_status()
                pos = float(last.tt_pos_deg)
                moving = bool(last.tt_in_motion)

                # Detect motion either via flag or via position change.
                if moving or abs(normalize_angle_deg(pos - start_pos)) >= tol_deg:
                    saw_motion = True

                # Done condition: not moving and close to expected.
                if (not moving) and abs(normalize_angle_deg(pos - expected)) <= tol_deg:
                    return

                if (not saw_motion) and (time.time() >= start_deadline) and (not moving) and abs(normalize_angle_deg(pos - expected)) > tol_deg:
                    raise TimeoutError("Turntable did not start moving (no motion observed).")

                time.sleep(float(poll_s))

            raise TimeoutError("Turntable move did not complete before timeout.")

    def turntable_jog_cw(self) -> None:
        seq = self._next_seq(O_TT_CMD_SEQ)
        self.write_reg(self.sv_base + O_TT_CMD_WORD, TT_CMD_JOG_CW)
        self.write_reg(self.sv_base + O_TT_CMD_SEQ, seq)
        self._wait_ack(O_TT_ACK_SEQ, seq, timeout_s=2.0)

    def turntable_jog_ccw(self) -> None:
        seq = self._next_seq(O_TT_CMD_SEQ)
        self.write_reg(self.sv_base + O_TT_CMD_WORD, TT_CMD_JOG_CCW)
        self.write_reg(self.sv_base + O_TT_CMD_SEQ, seq)
        self._wait_ack(O_TT_ACK_SEQ, seq, timeout_s=2.0)

    def turntable_reset_home(self) -> None:
        """Enter home-reset mode (turntable ENA disabled so user can rotate by hand)."""
        seq = self._next_seq(O_TT_CMD_SEQ)
        self.write_reg(self.sv_base + O_TT_CMD_WORD, TT_CMD_RESET_HOME)
        self.write_reg(self.sv_base + O_TT_CMD_SEQ, seq)
        self._wait_ack(O_TT_ACK_SEQ, seq, timeout_s=2.0)

    def turntable_set_home(self) -> None:
        """Set the current physical position as home (0°) and exit home-reset mode."""
        seq = self._next_seq(O_TT_CMD_SEQ)
        self.write_reg(self.sv_base + O_TT_CMD_WORD, TT_CMD_SET_HOME)
        self.write_reg(self.sv_base + O_TT_CMD_SEQ, seq)
        self._wait_ack(O_TT_ACK_SEQ, seq, timeout_s=2.0)

    # --- waiting utilities ---
    def wait_until(self, predicate, *, timeout_s: float = 10.0, poll_s: float = 0.2) -> EnvistaStatus:
        """Poll status until predicate(status) is True, else raise TimeoutError."""
        deadline = time.time() + float(timeout_s)
        last = self.read_status()
        while time.time() < deadline:
            if predicate(last):
                return last
            time.sleep(float(poll_s))
            last = self.read_status()
        raise TimeoutError("Condition not met before timeout")
