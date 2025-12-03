import sys
from pathlib import Path


def app_root() -> Path:
    """Return the directory that should hold writable app data.

    When frozen by PyInstaller, this resolves to the folder next to the
    executable. During development it resolves to the repo root.
    """
    if getattr(sys, "frozen", False):
        try:
            exe_dir = Path(sys.executable).resolve().parent
            if exe_dir.exists():
                return exe_dir
        except Exception:
            pass
    return Path(__file__).resolve().parents[1]


def user_file(name: str) -> Path:
    """Path to a user-facing file that should live beside the app."""
    return app_root() / name


def ensure_parent(path: Path) -> Path:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path
