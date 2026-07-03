from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from aiohttp import web
from PIL import Image, ImageOps

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("mediapipe is required. Install dependencies from requirements.txt") from exc


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    model_path: Path = Path("models/efficientdet_lite0.tflite")
    target_objects: list[str] = field(default_factory=lambda: ["cat"])
    score_threshold: float = 0.4
    max_results: int = -1
    max_request_bytes: int = 20 * 1024 * 1024


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
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self._detectors: dict[tuple[tuple[str, ...], float], Any] = {}
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            for detector in self._detectors.values():
                close = getattr(detector, "close", None)
                if close is not None:
                    close()
            self._detectors.clear()

    def _create_detector(self, targets: list[str], threshold: float) -> Any:
        BaseOptions = mp.tasks.BaseOptions
        ObjectDetector = mp.tasks.vision.ObjectDetector
        ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        kwargs: dict[str, Any] = {
            "base_options": BaseOptions(model_asset_path=str(self.config.model_path)),
            "running_mode": VisionRunningMode.IMAGE,
            "score_threshold": threshold,
            "max_results": self.config.max_results,
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
                f"{utc_stamp()} mediapipe-http: creating detector targets={targets!r} threshold={threshold}",
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


class MediaPipeHttpServer:
    def __init__(self, config: ServerConfig) -> None:
        self.config = config
        self.detectors = DetectorPool(config)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mediapipe")

    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=self.config.max_request_bytes)
        app["server"] = self
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/v1/analyze-frame", self.handle_analyze_frame)
        app.on_cleanup.append(self.cleanup)
        return app

    async def run_detection(self, image_bytes: bytes, targets: list[str], threshold: float) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.detectors.detect, image_bytes, targets, threshold)

    async def cleanup(self, app: web.Application) -> None:
        self.detectors.close()
        self._executor.shutdown(wait=True, cancel_futures=True)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def handle_analyze_frame(self, request: web.Request) -> web.Response:
        if not request.content_type.startswith("multipart/"):
            return web.json_response({"ok": False, "error": "multipart/form-data request body is required"}, status=400)

        try:
            message, image_bytes = await self.read_multipart_message(request)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)

        response, status = await self.handle_message(message, image_bytes)
        return web.json_response(response, status=status)

    async def read_multipart_message(self, request: web.Request) -> tuple[dict[str, Any], bytes]:
        reader = await request.multipart()
        message: dict[str, Any] = {}
        image_bytes: bytes | None = None

        async for part in reader:
            if part.name == "image":
                image_bytes = await part.read(decode=False)
                continue

            value = await part.text()
            if part.name == "config":
                try:
                    config = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid config JSON: {exc}") from exc
                if not isinstance(config, dict):
                    raise ValueError("config multipart field must be a JSON object")
                message.update(config)
                continue

            if part.name:
                message[part.name] = value

        if image_bytes is None:
            raise ValueError("image multipart field is required")
        if not image_bytes:
            raise ValueError("image multipart field must not be empty")
        return message, image_bytes

    async def handle_message(self, message: dict[str, Any], image_bytes: bytes) -> tuple[dict[str, Any], int]:
        request_id = message.get("request_id")
        targets_value = message.get("targets") or self.config.target_objects
        targets = parse_targets(targets_value)
        if targets is None:
            return {"ok": False, "request_id": request_id, "error": "targets must be a string list"}, 400

        try:
            threshold = float(message.get("score_threshold", self.config.score_threshold))
        except (TypeError, ValueError):
            return {"ok": False, "request_id": request_id, "error": "score_threshold must be a number"}, 400

        started = time.perf_counter()
        try:
            result = await self.run_detection(image_bytes, targets, threshold)
        except Exception as exc:  # pragma: no cover - safety net for long-running daemon
            return {"ok": False, "request_id": request_id, "error": repr(exc)}, 500

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return {
            "ok": True,
            "request_id": request_id,
            "frame_id": message.get("frame_id"),
            "timestamp": message.get("timestamp"),
            "targets": targets,
            "score_threshold": threshold,
            "processing_ms": elapsed_ms,
            **result,
        }, 200


def parse_targets(value: Any) -> list[str] | None:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return _csv(value)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    return None


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve MediaPipe object detection over HTTP")
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP address to bind. Defaults to 127.0.0.1")
    parser.add_argument("--port", type=int, default=8080, help="TCP port to bind. Defaults to 8080")
    parser.add_argument("--model-path", type=Path, required=True, help="MediaPipe .tflite model path")
    parser.add_argument(
        "--target-object",
        dest="target_object_list",
        action="append",
        default=None,
        help="Allowed object label. Repeat for multiple labels. Defaults to cat.",
    )
    parser.add_argument(
        "--target-objects",
        dest="target_objects_csv",
        type=_csv,
        default=None,
        help="Comma-separated allowed object labels. Overrides --target-object when set.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.4, help="Minimum detection score")
    parser.add_argument("--max-results", type=int, default=-1, help="Maximum MediaPipe detection results")
    parser.add_argument("--max-request-bytes", type=int, default=20 * 1024 * 1024, help="Maximum HTTP request size")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ServerConfig:
    target_objects = args.target_objects_csv or args.target_object_list or ["cat"]
    return ServerConfig(
        host=args.host,
        port=args.port,
        model_path=args.model_path.expanduser().resolve(),
        target_objects=target_objects,
        score_threshold=args.score_threshold,
        max_results=args.max_results,
        max_request_bytes=args.max_request_bytes,
    )


def main(argv: list[str] | None = None) -> None:
    config = config_from_args(parse_args(argv))
    if not config.model_path.exists():
        raise SystemExit(f"model_path does not exist: {config.model_path}")

    server = MediaPipeHttpServer(config)
    print(f"{utc_stamp()} mediapipe-http: listening on http://{config.host}:{config.port}", file=sys.stderr, flush=True)
    web.run_app(server.make_app(), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
