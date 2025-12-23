from __future__ import annotations

import threading
from typing import Iterable, Optional, Tuple

from PyQt5.QtWidgets import QMessageBox

from services import project_loader, solvision_manager, debug_log
from services.config import save_state, state


class ModelController:
    """
    Handles Detectron model lifecycle and persistence.
    Thread offloads are kept here so views stay responsive.
    """

    def __init__(self):
        pass

    # --- Loaders ------------------------------------------------------------
    def load_attachment(self, path: str, on_success=None, on_error=None) -> None:
        if not path:
            return
        debug_log.log(f"model_load: top start path={path}")

        def _run():
            try:
                project_loader.load_project(path)
                st = state()
                st.attachment_path = path
                st.last_project_path = path
                save_state()
                debug_log.log(f"model_load: top success path={path}")
                if on_success:
                    on_success(path)
            except Exception as ex:
                debug_log.log(f"model_load: top failed path={path} err={ex}")
                if on_error:
                    on_error(path, str(ex))

        threading.Thread(target=_run, daemon=True).start()

    def load_front(self, path: str, on_success=None, on_error=None) -> None:
        if not path:
            return
        debug_log.log(f"model_load: front start path={path}")

        def _run():
            try:
                solvision_manager.load_project_for("front", path, mode="exe")
                st = state()
                st.front_attachment_path = path
                st.last_front_project_path = path
                save_state()
                debug_log.log(f"model_load: front success path={path}")
                if on_success:
                    on_success(path)
            except Exception as ex:
                debug_log.log(f"model_load: front failed path={path} err={ex}")
                if on_error:
                    on_error(path, str(ex))

        threading.Thread(target=_run, daemon=True).start()

    def load_defect(self, path: str, on_success=None, on_error=None) -> None:
        if not path:
            return
        debug_log.log(f"model_load: defect start path={path}")

        def _run():
            try:
                solvision_manager.load_project_for("defect", path, mode="exe")
                st = state()
                st.defect_path = path
                st.last_defect_project_path = path
                save_state()
                debug_log.log(f"model_load: defect success path={path}")
                if on_success:
                    on_success(path)
            except Exception as ex:
                debug_log.log(f"model_load: defect failed path={path} err={ex}")
                if on_error:
                    on_error(path, str(ex))

        threading.Thread(target=_run, daemon=True).start()

    # --- Guards -------------------------------------------------------------
    def ensure_models_loaded(
        self,
        required: Iterable[str] = ("top",),
        *,
        parent=None,
        show_dialog: bool = False,
        log_callback=None,
    ) -> bool:
        missing = []
        try:
            for name in required:
                if solvision_manager.current_project_path_for(name) is None:
                    missing.append(name)
        except Exception:
            missing = list(required)

        if not missing:
            return True

        msg = f"Please load model(s): {', '.join(missing)} before running."
        if show_dialog:
            try:
                QMessageBox.information(parent, "Models Required", msg)
            except Exception:
                pass
        elif log_callback:
            try:
                log_callback(msg)
            except Exception:
                pass
        return False
