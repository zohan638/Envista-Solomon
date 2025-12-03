"""
Detectron2-backed model loader and inference helper.

Provides the same public function names the UI expects (load_project,
detect, detect_for, etc.) without any SolVision/.NET dependencies.
"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

from .config import state as _state

_predictors: Dict[str, DefaultPredictor] = {}  # e.g., {"top": pred, "front": pred2}
_model_paths: Dict[str, str] = {}
_initialized_error: Optional[str] = None
_log_cb: Optional[Callable[[str], None]] = None

# Default thresholds and class list (only used as a last-resort placeholder)
_DEFAULT_SCORE_THRESHOLD = 0.1
# Higher default for top (step 1) detections
_DEFAULT_SCORE_THRESHOLD_TOP = 0.8
CLASS_NAMES: List[str] = ["attachment"]
# Per-model class names; overwritten by model_final.json during loading.
_class_names_per_model: Dict[str, List[str]] = {
    "top": CLASS_NAMES,
    "front": CLASS_NAMES,
    "defect": ["defect"],
}


@dataclass
class ModelMeta:
    class_names: List[str]
    score_threshold: Optional[float] = None
    class_thresholds: Optional[Dict[int, float]] = None
    max_detections: Optional[int] = None
    min_size_test: Optional[int] = None
    max_size_test: Optional[int] = None
    class_colors: Optional[List[str]] = None
    class_colors: Optional[List[str]] = None


_model_meta: Dict[str, ModelMeta] = {}


def set_ui_logger(cb: Optional[Callable[[str], None]]):
    """Optional UI logger callback."""
    global _log_cb
    _log_cb = cb


def _dprint(*args):
    try:
        msg = " ".join(str(a) for a in args)
        print("[Detectron]", msg, flush=True)
        if _log_cb is not None:
            try:
                _log_cb("[Detectron] " + msg)
            except Exception:
                pass
    except Exception:
        pass


def _safe_read_json(path: Path) -> Optional[dict]:
    try:
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _dprint(f"Could not read metadata {path}: {exc}")
    return None


def _parse_class_names_from_model_final(data: Optional[dict]) -> List[str]:
    if not data:
        return []
    params = data.get("LearningParameter") or {}
    names: List[str] = []
    raw = params.get("class_names")
    if raw:
        names = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not names:
        items = data.get("ClassItems") or []
        names = [
            item.get("Name")
            for item in sorted(items, key=lambda i: i.get("ID", 0))
            if item.get("Name")
        ]
    return [n for n in names if n]


def _build_predictor(model_path: str, meta: ModelMeta) -> DefaultPredictor:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(meta.class_names)
    cfg.MODEL.RETINANET.NUM_CLASSES = len(meta.class_names)
    cfg.MODEL.WEIGHTS = model_path
    # Keep detectron threshold low; we filter manually per request
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.0
    if meta.max_detections is not None:
        cfg.TEST.DETECTIONS_PER_IMAGE = int(meta.max_detections)
    if meta.min_size_test is not None:
        cfg.INPUT.MIN_SIZE_TEST = int(meta.min_size_test)
    if meta.max_size_test is not None:
        cfg.INPUT.MAX_SIZE_TEST = int(meta.max_size_test)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DefaultPredictor(cfg)


def ensure_initialized() -> bool:
    """Detectron pipeline has no global init; always ready."""
    global _initialized_error
    _initialized_error = None
    return True


def initialization_error() -> Optional[str]:
    return _initialized_error


def _coerce_float(val, default: Optional[float]) -> Optional[float]:
    try:
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default


def _coerce_int(val, default: Optional[int]) -> Optional[int]:
    try:
        if val is None or val == "":
            return default
        return int(val)
    except Exception:
        return default


def _load_model_metadata(model_path: str) -> ModelMeta:
    path = Path(model_path).resolve()
    meta_dir = path.parent

    model_info = _safe_read_json(meta_dir / "model_final.json")
    if model_info is None:
        sidecar = path.with_suffix(".json")
        model_info = _safe_read_json(sidecar)
    if model_info is None:
        raise FileNotFoundError(
            f"Model metadata (model_final.json) not found next to checkpoint: {meta_dir}"
        )

    class_names = _parse_class_names_from_model_final(model_info)
    if not class_names:
        raise RuntimeError("No class names found in model_final.json")

    params = (model_info or {}).get("LearningParameter") or {}
    score_threshold = _coerce_float(params.get("test_score_thresh"), None)
    max_detections = _coerce_int(params.get("max_detections"), None)
    # Prefer explicit min/max dimension; fall back to recorded image dimensions.
    min_size_test = _coerce_int(params.get("min_dimension") or params.get("image_height"), None)
    max_size_test = _coerce_int(params.get("max_dimension") or params.get("image_width"), None)
    class_colors = None
    try:
        raw_colors = params.get("class_colors")
        if raw_colors:
            class_colors = [c.strip() for c in str(raw_colors).split(",") if c.strip()]
            if len(class_colors) != len(class_names):
                class_colors = None
    except Exception:
        class_colors = None
    if not class_colors:
        raise RuntimeError("No class_colors found in model_final.json")

    meta = ModelMeta(
        class_names=class_names,
        score_threshold=score_threshold,
        class_thresholds=None,
        max_detections=max_detections,
        min_size_test=min_size_test,
        max_size_test=max_size_test,
        class_colors=class_colors,
    )
    return meta


def load_project(path: str) -> None:
    """Load the default (top) model."""
    load_project_for("top", path)


def load_project_inproc(path: str) -> None:
    # Kept for backward compatibility with worker scripts (no-op difference).
    load_project(path)


def load_project_for(name: str, path: str, *, mode: str = "exe") -> None:
    """Load a model checkpoint for a given role (top/front/defect)."""
    global _initialized_error
    if not path:
        raise ValueError("Empty model path")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        meta = _load_model_metadata(path)
        _class_names_per_model[name] = meta.class_names
        _model_meta[name] = meta
        pred = _build_predictor(path, meta)
        _predictors[name] = pred
        _model_paths[name] = os.path.abspath(path)
        if name == "top":
            # Mirror legacy single-project tracking
            _model_paths["_default"] = _model_paths[name]
        _dprint(
            f"Loaded model for '{name}': {path} "
            f"({len(meta.class_names)} classes; threshold={meta.score_threshold})"
        )
    except Exception as exc:
        _initialized_error = str(exc)
        _dprint(f"Failed to load model '{name}': {exc}")
        raise


def has_loaded_project() -> bool:
    return "top" in _predictors


def current_project_path() -> Optional[str]:
    return _model_paths.get("top")


def current_project_path_for(name: str) -> Optional[str]:
    return _model_paths.get(name)


def class_colors_for(name: str) -> Optional[List[str]]:
    meta = _model_meta.get(name)
    if meta is None:
        raise RuntimeError(f"Model '{name}' metadata not loaded.")
    if not meta.class_colors:
        raise RuntimeError(f"Model '{name}' is missing class_colors in metadata.")
    return meta.class_colors


def diagnostics() -> Dict[str, Any]:
    return {
        "loaded": list(_predictors.keys()),
        "models": dict(_model_paths),
        "class_names": dict(_class_names_per_model),
        "score_thresholds": {k: v.score_threshold for k, v in _model_meta.items()},
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "initialized": True,
        "error": _initialized_error,
    }


def diagnostics_text() -> str:
    d = diagnostics()
    lines = [
        f"loaded: {d.get('loaded')}",
        f"models: {d.get('models')}",
        f"device: {d.get('device')}",
    ]
    if d.get("error"):
        lines.append(f"error: {d.get('error')}")
    return "\n".join(lines)


def _normalize_detections(
    instances,
    score_threshold: float,
    class_names: List[str],
    class_colors: Optional[List[str]] = None,
    per_class_thresholds: Optional[Dict[int, float]] = None,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    if instances is None:
        return results

    boxes = instances.pred_boxes if instances.has("pred_boxes") else None
    scores = instances.scores if instances.has("scores") else None
    classes = instances.pred_classes if instances.has("pred_classes") else None
    masks = instances.pred_masks if instances.has("pred_masks") else None

    num = len(instances)
    for i in range(num):
        sc = float(scores[i]) if scores is not None else 0.0
        cls_id = int(classes[i]) if classes is not None else None
        thr_for_class = score_threshold
        if per_class_thresholds is not None and cls_id is not None:
            thr_for_class = per_class_thresholds.get(cls_id, score_threshold)
        if thr_for_class is None:
            thr_for_class = 0.0
        if sc < thr_for_class:
            continue
        box = boxes[i].tensor.numpy().tolist()[0] if boxes is not None else None
        if box is None:
            continue
        x1, y1, x2, y2 = box
        w = x2 - x1
        h = y2 - y1
        cls_name = (
            class_names[cls_id]
            if cls_id is not None and 0 <= cls_id < len(class_names)
            else str(cls_id)
        )
        color_hex = None
        if (
            class_colors is not None
            and cls_id is not None
            and 0 <= cls_id < len(class_colors)
        ):
            color_hex = class_colors[cls_id]
        mask = masks[i].numpy() if masks is not None else None
        area = float(w * h) if w is not None and h is not None else None
        if mask is not None:
            try:
                area = float(mask.sum())
            except Exception:
                pass
        results.append(
            {
                "class": cls_name,
                "class_id": cls_id,
                "score": sc,
                "bounds": (float(x1), float(y1), float(w), float(h)),
                "area": area,
                "mask": mask,
                "rect": {"x": float(x1), "y": float(y1), "width": float(w), "height": float(h)},
                "color": color_hex,
            }
        )
    return results


def detect(image_path: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    """Run detection with the default (top) model."""
    return detect_for("top", image_path, score_threshold=score_threshold)


def detect_inproc(image_path: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    # Kept for compatibility; identical to detect().
    return detect(image_path, score_threshold=score_threshold)


def detect_for(name: str, image_path: str, score_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
    if not image_path:
        raise ValueError("Empty image path")
    if name not in _predictors:
        raise RuntimeError(f"Model '{name}' not loaded")
    meta = _model_meta.get(name)
    if meta is None:
        raise RuntimeError(f"Metadata not loaded for model '{name}'. Ensure model_final.json is present.")

    default_thr = (
        meta.score_threshold
        if meta is not None and meta.score_threshold is not None
        else (_DEFAULT_SCORE_THRESHOLD_TOP if name == "top" else _DEFAULT_SCORE_THRESHOLD)
    )
    # If colors are missing (e.g., model was loaded before color support), try to refresh metadata.
    if meta is not None and meta.class_colors is None:
        raise RuntimeError(f"Model '{name}' is missing class_colors; reload with valid metadata.")
    state_thr = getattr(_state(), "solvision_score_threshold", None)
    thr = _coerce_float(
        score_threshold,
        _coerce_float(state_thr, default_thr),
    )
    if thr is None:
        thr = 0.0

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    predictor = _predictors[name]
    class_names = (
        meta.class_names if meta is not None else _class_names_per_model.get(name, CLASS_NAMES)
    )
    per_class_thresholds = (
        meta.class_thresholds if (meta is not None and score_threshold is None and state_thr is None) else None
    )
    class_colors = class_colors_for(name)
    outputs = predictor(img)
    instances = outputs.get("instances", None)
    if instances is not None:
        instances = instances.to("cpu")

    return _normalize_detections(instances, thr, class_names, class_colors, per_class_thresholds)


def dispose():
    """Release predictors (best effort)."""
    _predictors.clear()
    _model_paths.clear()
    _model_meta.clear()
