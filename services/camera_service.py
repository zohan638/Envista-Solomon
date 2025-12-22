"""
Harvester/GenTL camera service.

This module intentionally exposes the same public API as the previous multi-backend
camera_service so downstream callers (UI + camera_manager) keep working:

- enumerate_devices() -> list[dict] with at least {index:int, name:str}
- connect(role:str, index:int) -> bool            # index is global (across devices)
- disconnect(role:str) -> None
- is_connected(role:str) -> bool
- get_connected_index(role:str) -> Optional[int]  # returns global index
- capture(role:str) -> numpy.ndarray (BGR, uint8)
- flush(role:str, frames:int=2, timeout_ms:int=50) -> None
- release_all() -> None
- backend_name() -> str
- diagnostics() -> dict
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import os
import re
import socket
import struct
import threading
import time
from typing import Any, Dict, List, Optional

from .app_paths import app_root

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore

try:
    import cv2 as _cv2  # type: ignore
except Exception:
    _cv2 = None  # type: ignore

try:
    from genicam.gentl import TimeoutException  # type: ignore
    from harvesters.core import Harvester  # type: ignore
except Exception as ex:  # pragma: no cover - runtime/environment specific
    Harvester = None  # type: ignore
    TimeoutException = None  # type: ignore
    _import_error = ex
else:
    _import_error = None


# Default bundled producer (repo/app relative).
DEFAULT_GENTL_RELATIVE = os.path.join("MV GigE V", "MVProducerGEV.cti")

# Default vendor installation path (if MV Viewer is installed).
DEFAULT_VENDOR_GENTL = r"C:\Program Files\HuarayTech\MV Viewer\Runtime\x64\MVProducerGEV.cti"

# Only open GigE Vision cameras.
GIGE_TL_TYPES = {"GEV", "GIGE", "GIGEVISION"}

# Default capture timeout for capture() (milliseconds).
DEFAULT_CAPTURE_TIMEOUT_MS = 1500


_IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")


def _normalize_role(role: str) -> str:
    return "Top" if str(role) == "Top" else "Front"


def _format_ip_from_int(value: int) -> str:
    ip_be = socket.inet_ntoa(struct.pack("!I", int(value)))
    ip_le = socket.inet_ntoa(struct.pack("<I", int(value)))
    if ip_be != "0.0.0.0":
        return ip_be
    return ip_le


def _get_device_ip(node_map) -> Optional[str]:
    for name in ("GevCurrentIPAddress", "GevDeviceIPAddress", "GevIPAddress"):
        node = getattr(node_map, name, None)
        if node is None:
            continue
        try:
            value = node.value
        except Exception:
            continue
        if isinstance(value, str):
            return value
        if isinstance(value, bytes) and len(value) == 4:
            try:
                return socket.inet_ntoa(value)
            except OSError:
                continue
        if isinstance(value, int):
            return _format_ip_from_int(value)
    return None


def _is_gige_device_info(info) -> bool:
    tl_type = (getattr(info, "tl_type", "") or "").upper()
    if not tl_type:
        # Some GenTL producers don't populate tl_type; assume True when unknown.
        return True
    if tl_type in GIGE_TL_TYPES:
        return True
    if any(token in tl_type for token in GIGE_TL_TYPES):
        return True
    for attr in ("display_name", "id_", "model", "vendor", "user_defined_name"):
        value = getattr(info, attr, None)
        if not value:
            continue
        text = str(value).upper()
        if any(token in text for token in GIGE_TL_TYPES):
            return True
    return False


def _extract_ip(text: str) -> Optional[str]:
    if not text:
        return None
    m = _IP_RE.search(text)
    if not m:
        return None
    return m.group(0)


def _reshape_or_fallback(data, *shape):
    """Best-effort reshape that tolerates padded line strides.

    Some producers expose buffers with per-line padding, so the flat array size
    is larger than the expected image size.
    """
    try:
        return data.reshape(*shape)
    except Exception:
        pass

    if np is None:
        return data

    try:
        arr = np.asarray(data)
    except Exception:
        return data

    if arr.ndim != 1 or len(shape) not in (2, 3):
        return arr

    try:
        h = int(shape[0])
        w = int(shape[1])
    except Exception:
        return arr
    if h <= 0 or w <= 0:
        return arr

    try:
        if arr.size % h != 0:
            return arr
        row_elems = int(arr.size // h)
    except Exception:
        return arr

    if len(shape) == 2:
        if row_elems < w:
            return arr
        try:
            return arr.reshape(h, row_elems)[:, :w]
        except Exception:
            return arr

    # len(shape) == 3
    try:
        c = int(shape[2])
    except Exception:
        return arr
    if c <= 0:
        return arr
    needed = w * c
    if row_elems < needed:
        return arr
    try:
        tmp = arr.reshape(h, row_elems)[:, :needed]
        return tmp.reshape(h, w, c)
    except Exception:
        return arr


def _try_set_node_value(node_map, name: str, value) -> bool:
    node = getattr(node_map, name, None)
    if node is None:
        return False
    try:
        if hasattr(node, "is_writable") and not node.is_writable:
            return False
        node.value = value
        return True
    except Exception:
        return False


def _try_execute_node(node_map, name: str) -> bool:
    node = getattr(node_map, name, None)
    if node is None:
        return False
    try:
        if hasattr(node, "is_writable") and not node.is_writable:
            return False
        execute = getattr(node, "execute", None)
        if callable(execute):
            execute()
            return True
        if callable(node):
            node()
            return True
    except Exception:
        return False
    return False


def _configure_streaming(node_map) -> None:
    _try_set_node_value(node_map, "AcquisitionMode", "Continuous")
    _try_set_node_value(node_map, "TriggerSelector", "FrameStart")
    _try_set_node_value(node_map, "TriggerMode", "Off")
    # For mono cameras/producers that default to packed 10/12-bit formats,
    # force an 8-bit format when supported so preview works reliably.
    _try_set_node_value(node_map, "PixelFormat", "Mono8")


def _ms_to_s(value_ms: int) -> float:
    try:
        ms = float(value_ms)
    except Exception:
        ms = float(DEFAULT_CAPTURE_TIMEOUT_MS)
    return max(0.001, ms / 1000.0)


def _add_dll_dir(path: str) -> None:
    if not path:
        return
    try:
        if not os.path.isdir(path):
            return
    except Exception:
        return
    try:
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(path)  # type: ignore[attr-defined]
        else:
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def _configure_runtime_paths(gentl_file: Path) -> None:
    # Ensure the GenTL producer's folder is on the DLL search path.
    try:
        _add_dll_dir(str(gentl_file.parent))
    except Exception:
        pass
    # Best-effort: also add default MV Viewer runtime paths (if installed).
    rt_dir = Path(DEFAULT_VENDOR_GENTL).parent
    candidates = [
        rt_dir,
        rt_dir / "GenICam" / "bin" / "Win64_x64",
        rt_dir / "GenICam" / "bin64",
        rt_dir / "GenICam" / "bin" / "win64_x64",
    ]
    for p in candidates:
        try:
            _add_dll_dir(str(p))
        except Exception:
            pass


def _resolve_gentl_file() -> Optional[Path]:
    """Resolve the GenTL Producer (.cti) to use.

    Priority:
    1) ENVISTA_GENTL_FILE or GENTL_FILE (absolute or relative to app_root)
    2) Repo/app relative MV GigE V/MVProducerGEV.cti
    3) Default MV Viewer installation path
    """
    candidates: List[Path] = []

    env = os.getenv("ENVISTA_GENTL_FILE") or os.getenv("GENTL_FILE")
    if env:
        p = Path(env).expanduser()
        candidates.append(p if p.is_absolute() else (app_root() / p))
        candidates.append(p)

    candidates.append(app_root() / DEFAULT_GENTL_RELATIVE)
    candidates.append(Path(DEFAULT_VENDOR_GENTL))

    seen = set()
    for p in candidates:
        try:
            ps = str(p)
        except Exception:
            continue
        if ps in seen:
            continue
        seen.add(ps)
        try:
            if p.is_file():
                return p.resolve()
        except Exception:
            continue
    return None


def _import_ok() -> bool:
    return bool(Harvester is not None and TimeoutException is not None and np is not None)


_diag: Dict[str, object] = {
    "import_ok": _import_ok(),
    "load_error": None if _import_error is None else str(_import_error),
    "gentl_file": DEFAULT_GENTL_RELATIVE,
    "gentl_file_resolved": None,
    "dev_num": None,
    "gige_dev_num": None,
    "last_error": None,
    "last_update_s": None,
}


@dataclass
class _RoleCtx:
    ia: Any
    index: int  # global index
    backend_index: int  # harvester device index
    label: str
    lock: threading.RLock
    needs_acquisition_start: bool = False


_LOCK = threading.RLock()
_HARVESTER: Optional[Any] = None
_DEVICES: List[Dict[str, object]] = []
_ROLE_CONN: Dict[str, Optional[_RoleCtx]] = {"Top": None, "Front": None}


def _has_active_connections() -> bool:
    try:
        return any(_ROLE_CONN.get(r) is not None for r in ("Top", "Front"))
    except Exception:
        return False


def _ensure_harvester() -> Optional[Any]:
    global _HARVESTER
    if _HARVESTER is not None:
        return _HARVESTER
    if not _import_ok():
        _diag["import_ok"] = False
        _diag["last_error"] = "harvesters/genicam/numpy not available"
        return None

    gentl_file = _resolve_gentl_file()
    _diag["gentl_file_resolved"] = str(gentl_file) if gentl_file else None
    if gentl_file is None:
        _diag["last_error"] = (
            "GenTL producer (.cti) not found. Set ENVISTA_GENTL_FILE or place "
            f"'{DEFAULT_GENTL_RELATIVE}' next to the app."
        )
        return None

    _configure_runtime_paths(gentl_file)
    try:
        with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
            h = Harvester()
            h.add_file(str(gentl_file))
            h.update()
        _HARVESTER = h
        _diag["last_error"] = None
        _diag["last_update_s"] = time.time()
        return _HARVESTER
    except Exception as ex:  # pragma: no cover - runtime/environment specific
        _diag["last_error"] = str(ex)
        try:
            h.reset()
        except Exception:
            pass
        return None


def backend_name() -> str:
    return "HARVESTERS" if _import_ok() else "None"


def _rebuild_device_table(max_devices: int = 8) -> None:
    global _DEVICES
    devices: List[Dict[str, object]] = []

    h = _ensure_harvester()
    if h is None:
        _DEVICES = []
        return

    # IMPORTANT:
    # Calling Harvester.update() while ImageAcquirers are active can invalidate
    # running streams (and, in some environments, crash). Only refresh the
    # device list when no cameras are currently connected.
    if not _has_active_connections():
        try:
            with open(os.devnull, "w") as _devnull, contextlib.redirect_stdout(_devnull), contextlib.redirect_stderr(_devnull):
                h.update()
            _diag["last_update_s"] = time.time()
        except Exception as ex:
            _diag["last_error"] = str(ex)
            _DEVICES = []
            return

    try:
        info_list = list(getattr(h, "device_info_list", []) or [])
    except Exception:
        info_list = []

    _diag["dev_num"] = len(info_list)

    gige = [(i, info) for i, info in enumerate(info_list) if _is_gige_device_info(info)]
    _diag["gige_dev_num"] = len(gige)

    # If filtering ended up hiding everything but devices exist, show all as a fallback.
    if not gige and info_list:
        gige = list(enumerate(info_list))

    for global_index, (backend_index, info) in enumerate(gige[: max(0, int(max_devices))]):
        base = (
            getattr(info, "display_name", None)
            or getattr(info, "user_defined_name", None)
            or getattr(info, "model", None)
            or getattr(info, "id_", None)
            or f"Device {backend_index}"
        )
        base_str = str(base)
        ip = (
            _extract_ip(base_str)
            or _extract_ip(str(getattr(info, "id_", "") or ""))
            or _extract_ip(str(getattr(info, "model", "") or ""))
        )
        name = f"{base_str} ({ip})" if ip else base_str
        devices.append(
            {
                "index": int(global_index),
                "name": f"{name} [HARVESTERS]",
                "backend": "HARVESTERS",
                "backend_index": int(backend_index),
            }
        )

    _DEVICES = devices


def enumerate_devices(max_devices: int = 8) -> List[Dict]:
    with _LOCK:
        _rebuild_device_table(max_devices=max_devices)
        return list(_DEVICES)


def _find_device(global_index: int) -> Optional[Dict[str, object]]:
    with _LOCK:
        if not _DEVICES:
            _rebuild_device_table()
        for d in _DEVICES:
            try:
                if int(d.get("index", -1)) == int(global_index):
                    return d
            except Exception:
                continue
        # Retry after a rebuild (devices may have changed).
        _rebuild_device_table()
        for d in _DEVICES:
            try:
                if int(d.get("index", -1)) == int(global_index):
                    return d
            except Exception:
                continue
        return None


def _stop_and_destroy(ia: Any) -> None:
    if ia is None:
        return
    try:
        try:
            ia.stop()
        except Exception:
            pass
        try:
            ia.destroy()
        except Exception:
            pass
    except Exception:
        pass


def connect(role: str, index: int) -> bool:
    role = _normalize_role(role)
    with _LOCK:
        dev = _find_device(index)
        if not dev:
            return False

        try:
            backend_index = int(dev.get("backend_index", -1))
        except Exception:
            backend_index = -1
        if backend_index < 0:
            return False

        # Prevent a single physical device from being assigned twice.
        other = "Front" if role == "Top" else "Top"
        other_ctx = _ROLE_CONN.get(other)
        if other_ctx is not None and int(other_ctx.backend_index) == int(backend_index):
            return False

        h = _ensure_harvester()
        if h is None:
            return False

        # Disconnect any previous camera for this role.
        disconnect(role)

        ia = None
        try:
            ia = h.create(int(backend_index))
            # Configure acquisition parameters (best-effort).
            try:
                _configure_streaming(ia.remote_device.node_map)
            except Exception:
                pass

            label = str(dev.get("name") or f"Device {index}")
            # Prefer IP as a short label if available.
            try:
                ip = _get_device_ip(ia.remote_device.node_map)
                if ip:
                    label = ip
            except Exception:
                pass

            try:
                ia.start()
            except Exception as ex:
                _diag["last_error"] = str(ex)
                _stop_and_destroy(ia)
                return False

            # Some producers/cameras require explicitly executing AcquisitionStart
            # even after ia.start(); do it best-effort.
            try:
                _try_execute_node(ia.remote_device.node_map, "AcquisitionStart")
            except Exception:
                pass

            _ROLE_CONN[role] = _RoleCtx(
                ia=ia,
                index=int(index),
                backend_index=int(backend_index),
                label=label,
                lock=threading.RLock(),
            )
            _diag["last_error"] = None
            return True
        except Exception as ex:  # pragma: no cover - runtime/hardware specific
            _diag["last_error"] = str(ex)
            _stop_and_destroy(ia)
            return False


def disconnect(role: str) -> None:
    role = _normalize_role(role)
    with _LOCK:
        ctx = _ROLE_CONN.get(role)
        if ctx is None:
            return
        try:
            with ctx.lock:
                _stop_and_destroy(ctx.ia)
        finally:
            _ROLE_CONN[role] = None


def is_connected(role: str) -> bool:
    role = _normalize_role(role)
    with _LOCK:
        return _ROLE_CONN.get(role) is not None


def get_connected_index(role: str) -> Optional[int]:
    role = _normalize_role(role)
    with _LOCK:
        ctx = _ROLE_CONN.get(role)
        if ctx is None:
            return None
        try:
            return int(ctx.index)
        except Exception:
            return None


def _to_uint8_channel(channel):
    if np is None:
        return channel
    try:
        if channel.dtype == np.uint8:
            return channel
    except Exception:
        return channel

    arr = channel.astype(np.float32, copy=False)
    try:
        mn = float(arr.min())
        mx = float(arr.max())
    except Exception:
        return channel.astype(np.uint8, copy=False)
    if mx - mn < 1e-6:
        # Constant image: preserve brightness instead of forcing black.
        try:
            if np.issubdtype(channel.dtype, np.integer):
                bits = int(getattr(channel.dtype, "itemsize", 1) or 1) * 8
                v = int(channel.flat[0])
                if bits > 8:
                    v = v >> max(0, bits - 8)
                v = int(max(0, min(255, v)))
                return np.full(channel.shape, v, dtype=np.uint8)
            if np.issubdtype(channel.dtype, np.floating):
                v = float(channel.flat[0])
                if v <= 1.0:
                    v = v * 255.0
                v = int(max(0, min(255, int(round(v)))))
                return np.full(channel.shape, v, dtype=np.uint8)
        except Exception:
            pass
        return np.zeros(channel.shape, dtype=np.uint8)
    norm = (arr - mn) / (mx - mn)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


_BAYER_TO_BGR = {
    "BAYERBG": getattr(_cv2, "COLOR_BayerBG2BGR", None),
    "BAYERGB": getattr(_cv2, "COLOR_BayerGB2BGR", None),
    "BAYERGR": getattr(_cv2, "COLOR_BayerGR2BGR", None),
    "BAYERRG": getattr(_cv2, "COLOR_BayerRG2BGR", None),
}


def _mono_to_bgr(gray):
    if np is None:
        return gray
    if _cv2 is not None:
        return _cv2.cvtColor(gray, _cv2.COLOR_GRAY2BGR)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _try_decode_mono_packed(raw, height: int, width: int, data_format: str):
    """Decode common packed mono formats when producer exposes raw bytes.

    Returns a 2D uint8 image on success, else None.
    """
    if np is None:
        return None
    try:
        a = np.asarray(raw)
    except Exception:
        return None
    if a.ndim != 1 or height <= 0 or width <= 0:
        return None

    fmt = (data_format or "").upper()

    # Mono16 provided as bytes
    if a.dtype == np.uint8 and ("MONO16" in fmt or "16" in fmt):
        try:
            if a.size % 2 != 0:
                return None
            u16 = a.view(np.uint16)
            img16 = _reshape_or_fallback(u16, height, width)
            img16 = np.asarray(img16)
            if img16.ndim != 2:
                return None
            return (img16 >> 8).astype(np.uint8, copy=False)
        except Exception:
            return None

    # Mono12 packed (most common 12-bit packing: 3 bytes -> 2 pixels)
    if a.dtype == np.uint8 and ("MONO12" in fmt or "12" in fmt):
        pixels = int(height * width)
        if pixels <= 0:
            return None
        pair_pixels = pixels // 2
        triplets = pair_pixels * 3
        if a.size < triplets:
            return None
        try:
            b = a[:triplets].reshape(-1, 3).astype(np.uint16, copy=False)
            # Variant A (LSB-first): p0= b0 + (b1&0x0F)<<8, p1= (b2<<4) + (b1>>4)
            p0a = b[:, 0] | ((b[:, 1] & 0x0F) << 8)
            p1a = (b[:, 2] << 4) | (b[:, 1] >> 4)
            # Variant B: p0= (b0<<4) + (b1>>4), p1= ((b1&0x0F)<<8) + b2
            p0b = (b[:, 0] << 4) | (b[:, 1] >> 4)
            p1b = ((b[:, 1] & 0x0F) << 8) | b[:, 2]

            img_a = np.empty(pixels, dtype=np.uint16)
            img_b = np.empty(pixels, dtype=np.uint16)
            img_a[0 : 2 * pair_pixels : 2] = p0a
            img_a[1 : 2 * pair_pixels : 2] = p1a
            img_b[0 : 2 * pair_pixels : 2] = p0b
            img_b[1 : 2 * pair_pixels : 2] = p1b

            # Choose the decoding that yields more variance (heuristic).
            try:
                var_a = float(img_a.var())
                var_b = float(img_b.var())
            except Exception:
                var_a = 0.0
                var_b = 0.0
            img = img_a if var_a >= var_b else img_b

            # Optional last pixel (odd pixel count): decode from 2 remaining bytes.
            if pixels % 2 == 1 and a.size >= triplets + 2:
                b0 = int(a[triplets])
                b1 = int(a[triplets + 1])
                if img is img_a:
                    last = b0 | ((b1 & 0x0F) << 8)
                else:
                    last = (b0 << 4) | (b1 >> 4)
                img[-1] = int(last)

            img = img.reshape(height, width)
            return (img >> 4).astype(np.uint8, copy=False)
        except Exception:
            return None

    return None


def _component_to_bgr(component):
    if np is None:
        raise RuntimeError("numpy not available")

    data = getattr(component, "data", None)
    if data is None:
        raise RuntimeError("empty frame")
    try:
        if hasattr(data, "size") and int(data.size) == 0:  # type: ignore[truthy-bool]
            raise RuntimeError("empty frame")
    except Exception:
        pass

    try:
        height = int(getattr(component, "height", 0) or 0)
        width = int(getattr(component, "width", 0) or 0)
    except Exception:
        height = 0
        width = 0
    if height <= 0 or width <= 0:
        raise RuntimeError("invalid frame dimensions")

    try:
        num_components = int(getattr(component, "num_components_per_pixel", 1) or 1)
    except Exception:
        num_components = 1
    data_format = (getattr(component, "data_format", "") or "").upper()

    # --- Reshape to an image ---
    if num_components == 1:
        decoded = None
        try:
            decoded = _try_decode_mono_packed(data, height, width, data_format)
        except Exception:
            decoded = None
        if decoded is not None:
            frame = decoded
        else:
            frame = _reshape_or_fallback(data, height, width)
        # Demosaic Bayer to BGR when possible.
        if "BAYER" in data_format:
            if _cv2 is None:
                raise RuntimeError("OpenCV not available for Bayer conversion")
            for key, code in _BAYER_TO_BGR.items():
                if key in data_format and code is not None:
                    frame = _cv2.cvtColor(frame, code)
                    break
        if getattr(frame, "ndim", 0) == 2:
            # If the producer provides higher bit-depth mono formats as uint16,
            # downshift to 8-bit for display.
            try:
                f = np.asarray(frame)
                if f.dtype == np.uint16 and ("MONO12" in data_format or "12" in data_format):
                    frame = (f >> 4).astype(np.uint8, copy=False)
                elif f.dtype == np.uint16 and ("MONO16" in data_format or "16" in data_format):
                    frame = (f >> 8).astype(np.uint8, copy=False)
            except Exception:
                pass
            frame = _mono_to_bgr(frame)

    elif num_components == 3:
        frame = _reshape_or_fallback(data, height, width, 3)
        if "RGB" in data_format and "BGR" not in data_format:
            frame = frame[:, :, ::-1]

    elif num_components == 4:
        frame = _reshape_or_fallback(data, height, width, 4)
        if _cv2 is not None:
            if "RGBA" in data_format:
                frame = _cv2.cvtColor(frame, _cv2.COLOR_RGBA2BGR)
            elif "BGRA" in data_format:
                frame = _cv2.cvtColor(frame, _cv2.COLOR_BGRA2BGR)
            else:
                frame = frame[:, :, :3]
        else:
            frame = frame[:, :, :3]

    else:
        frame = _reshape_or_fallback(data, height, width, num_components)
        if getattr(frame, "ndim", 0) >= 3 and frame.shape[2] >= 3:  # type: ignore[index]
            frame = frame[:, :, :3]
        elif getattr(frame, "ndim", 0) == 2:
            frame = _mono_to_bgr(frame)

    # --- Normalize dtype to uint8 BGR ---
    frame = np.asarray(frame)
    if frame.ndim == 2:
        frame = _mono_to_bgr(_to_uint8_channel(frame))
    elif frame.ndim == 3 and frame.shape[2] >= 3:
        if frame.dtype != np.uint8:
            ch = [_to_uint8_channel(frame[:, :, c]) for c in range(3)]
            frame = np.stack(ch, axis=2)
        frame = frame[:, :, :3]
    else:
        frame = frame.astype(np.uint8, copy=False)
        if frame.ndim == 2:
            frame = _mono_to_bgr(frame)

    # Always copy because Harvester buffers get released after fetch().
    return np.array(frame, copy=True)


def capture(role: str, timeout_ms: int = DEFAULT_CAPTURE_TIMEOUT_MS):
    role = _normalize_role(role)
    with _LOCK:
        ctx = _ROLE_CONN.get(role)
        if ctx is None:
            raise RuntimeError("camera not connected")

    with ctx.lock:
        ia = ctx.ia
        if ia is None:
            raise RuntimeError("camera not connected")

        try:
            timeout_s = _ms_to_s(timeout_ms)

            def _fetch_once():
                with ia.fetch(timeout=timeout_s) as buffer:
                    if buffer is None:
                        raise RuntimeError("empty buffer")
                    component = buffer.payload.components[0]
                    return _component_to_bgr(component)

            # If we previously detected that the producer stops acquisition after
            # a single frame, pre-kick acquisition to avoid waiting for a timeout.
            if getattr(ctx, "needs_acquisition_start", False):
                try:
                    _try_execute_node(ia.remote_device.node_map, "AcquisitionStart")
                except Exception:
                    pass

            try:
                frame = _fetch_once()
            except Exception as ex:
                # If acquisition stopped after a single frame, try to kick it
                # back on and retry once.
                if TimeoutException is not None and isinstance(ex, TimeoutException):
                    try:
                        _try_execute_node(ia.remote_device.node_map, "AcquisitionStart")
                    except Exception:
                        pass
                    frame = _fetch_once()
                    try:
                        ctx.needs_acquisition_start = True
                    except Exception:
                        pass
                else:
                    raise
        except Exception as ex:
            if TimeoutException is not None and isinstance(ex, TimeoutException):
                raise TimeoutError(f"camera fetch timeout after {int(timeout_ms)} ms") from ex
            raise

        # Hard-coded orientation adjustment: rotate Top camera 90 degrees clockwise.
        if role == "Top" and _cv2 is not None:
            try:
                frame = _cv2.rotate(frame, _cv2.ROTATE_90_CLOCKWISE)
            except Exception:
                pass
        return frame


def flush(role: str, frames: int = 2, timeout_ms: int = 50) -> None:
    role = _normalize_role(role)
    with _LOCK:
        ctx = _ROLE_CONN.get(role)
        if ctx is None:
            return

    with ctx.lock:
        ia = ctx.ia
        if ia is None:
            return
        timeout_s = _ms_to_s(timeout_ms)
        for _ in range(max(0, int(frames))):
            try:
                with ia.fetch(timeout=timeout_s) as _buffer:
                    pass
            except Exception as ex:
                if TimeoutException is not None and isinstance(ex, TimeoutException):
                    break
                break


def release_all() -> None:
    global _HARVESTER
    with _LOCK:
        for role in ("Top", "Front"):
            try:
                disconnect(role)
            except Exception:
                pass
        try:
            if _HARVESTER is not None:
                _HARVESTER.reset()
        except Exception:
            pass
        _HARVESTER = None
        _DEVICES.clear()


def diagnostics() -> Dict:
    return {
        "backends": ["HARVESTERS"] if _import_ok() else [],
        "devices": list(_DEVICES),
        "HARVESTERS": dict(_diag),
        "connected": {
            "Top": get_connected_index("Top"),
            "Front": get_connected_index("Front"),
        },
    }
