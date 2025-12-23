import sys
from services import solvision_manager
from services import crash_reporter


def _bootstrap_models():
    """Initialize Detectron models before Qt spins up (placeholder)."""
    if solvision_manager.ensure_initialized():
        return True
    err = solvision_manager.initialization_error() or "Model initialization failure."
    print(f"[Detectron] Failed to initialize before UI launch:\n{err}", file=sys.stderr, flush=True)
    return False


def main():
    # Install crash logging as early as possible so silent exits are captured.
    try:
        crash_reporter.install()
    except Exception:
        pass

    # Initialize detectron before importing any PyQt/UI modules.
    if not _bootstrap_models():
        return 1

    from PyQt5.QtWidgets import QApplication
    from controllers.app_controller import AppController

    app = QApplication(sys.argv)
    controller = AppController()
    window = controller.launch()
    if window is None:
        solvision_manager.dispose()
        return 0
    rc = app.exec_()
    solvision_manager.dispose()
    return rc


if __name__ == "__main__":
    sys.exit(main())
