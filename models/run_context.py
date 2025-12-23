from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunContext:
    """Per-inspection bookkeeping."""

    part_id_raw: str
    part_id_clean: str
    capture_dir: Optional[Path]
    cycle_start_ts: float

