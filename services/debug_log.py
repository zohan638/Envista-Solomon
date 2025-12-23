from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

from .app_paths import app_root

_log_lock = threading.Lock()
_log_path = app_root() / "debug.log"


def log(message: str) -> None:
    """Append a timestamped, thread-tagged message to debug.log (best effort)."""
    try:
        line = f"{datetime.now().isoformat()} [{threading.current_thread().name}] {message}\n"
        with _log_lock:
            _log_path.parent.mkdir(parents=True, exist_ok=True)
            _log_path.write_text("", encoding="utf-8") if not _log_path.exists() else None
            with _log_path.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        pass
