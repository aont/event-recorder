from __future__ import annotations

import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .ffmpeg_utils import utc_stamp

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("mediapipe is required. Install dependencies from requirements.txt") from exc


_GLOBAL_DETECTORS: "DetectorPool | None" = None


def category_name(category: Any) -> str:
    return getattr(category, "category_name", None) or getattr(category, "display_name", None) or ""


def bbox_to_dict(bbox: Any) -> dict[str, int]:
    return {
        "x": int(bbox.origin_x),
        "y": int(bbox.origin_y),
        "width": int(bbox.width),
        "height": int(bbox.height),
    }


def load_image_bytes_as_mediapipe_srgb(image_bytes: bytes) -> Any:
    with Image.open(BytesIO(image_bytes)) as pil_image:
        pil_image = ImageOps.exif_transpose(pil_image)
        pil_image = pil_image.convert("RGB")
        rgb_array = np.asarray(pil_image, dtype=np.uint8)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(rgb_array),
    )


class DetectorPool:
    def __init__(self, *, model_path: Path, max_results: int) -> None:
        self.model_path = model_path
        self.max_results = max_results
        self._detectors: dict[tuple[tuple[str, ...], float], Any] = {}
        self._lock = threading.Lock()

    def _create_detector(self, targets: list[str], threshold: float) -> Any:
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        kwargs: dict[str, Any] = {
            "base_options": BaseOptions(model_asset_path=str(self.model_path)),
            "running_mode": VisionRunningMode.IMAGE,
            "score_threshold": threshold,
            "max_results": self.max_results,
        }
        if targets:
            kwargs["category_allowlist"] = targets
        options = ObjectDetectorOptions(**kwargs)
        return ObjectDetector.create_from_options(options)

    def _get_detector(self, targets: list[str], threshold: float) -> Any:
        key = (tuple(targets), float(threshold))
        detector = self._detectors.get(key)
        if detector is None:
            print(
                f"{utc_stamp()} ai-worker: creating detector targets={targets!r} threshold={threshold}",
                file=sys.stderr,
                flush=True,
            )
            detector = self._create_detector(targets, threshold)
            self._detectors[key] = detector
        return detector

    def detect(self, image_bytes: bytes, targets: list[str], threshold: float) -> dict[str, Any]:
        mp_image = load_image_bytes_as_mediapipe_srgb(image_bytes)
        target_set = set(targets)
        detections: list[dict[str, Any]] = []

        with self._lock:
            detector = self._get_detector(targets, threshold)
            result = detector.detect(mp_image)

        for detection in result.detections:
            bbox = bbox_to_dict(detection.bounding_box)
            for category in detection.categories:
                name = category_name(category)
                score = float(category.score)
                if score < threshold:
                    continue
                if target_set and name not in target_set:
                    continue
                detections.append({"label": name, "score": score, "bbox": bbox})

        return {
            "has_target": len(detections) > 0,
            "num_targets": len(detections),
            "detections": detections,
        }


def init_ai_worker(model_path: str, max_results: int) -> None:
    global _GLOBAL_DETECTORS
    model = Path(model_path)
    if not model.exists():
        raise FileNotFoundError(f"model_path does not exist: {model}")
    _GLOBAL_DETECTORS = DetectorPool(model_path=model, max_results=max_results)


def analyze_frame_in_worker(image_bytes: bytes, targets: list[str], threshold: float) -> dict[str, Any]:
    if _GLOBAL_DETECTORS is None:
        raise RuntimeError("AI worker was not initialized")
    started = time.perf_counter()
    result = _GLOBAL_DETECTORS.detect(image_bytes, targets, threshold)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {"processing_ms": elapsed_ms, **result}
