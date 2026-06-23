from __future__ import annotations

import argparse
import asyncio
import base64
import sys
import threading
import time
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from .config import AppConfig, Proc2Config
from .ffmpeg_utils import utc_stamp
from .ipc import IpcProtocolError, read_json_message, write_json_message

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit("mediapipe is required for proc2. Install dependencies from requirements.txt") from exc


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
    def __init__(self, config: Proc2Config) -> None:
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
                f"{utc_stamp()} proc2: creating detector targets={targets!r} threshold={threshold}",
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


class Proc2Server:
    def __init__(self, config: Proc2Config) -> None:
        self.config = config
        self.detectors = DetectorPool(config)
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        if not self.config.model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {self.config.model_path}")
        self.config.socket_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.config.socket_path.unlink()
        except FileNotFoundError:
            pass
        self._server = await asyncio.start_unix_server(self.handle_client, path=str(self.config.socket_path))
        print(f"{utc_stamp()} proc2: listening on {self.config.socket_path}", file=sys.stderr, flush=True)

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        self.detectors.close()
        try:
            self.config.socket_path.unlink()
        except FileNotFoundError:
            pass

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                message = await read_json_message(reader, max_bytes=self.config.max_message_bytes)
                if message is None:
                    return
                response = await self.handle_message(message)
                await write_json_message(writer, response)
        except IpcProtocolError as exc:
            await write_json_message(writer, {"ok": False, "error": f"protocol_error: {exc}"})
        except Exception as exc:  # pragma: no cover - safety net for long-running daemon
            try:
                await write_json_message(writer, {"ok": False, "error": repr(exc)})
            except Exception:
                pass
        finally:
            writer.close()
            await writer.wait_closed()

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        if message.get("type") != "analyze_frame":
            return {"ok": False, "request_id": message.get("request_id"), "error": "unknown message type"}

        request_id = message.get("request_id")
        targets = message.get("targets") or self.config.target_objects
        if not isinstance(targets, list) or not all(isinstance(x, str) for x in targets):
            return {"ok": False, "request_id": request_id, "error": "targets must be a string list"}
        threshold = float(message.get("score_threshold", self.config.score_threshold))

        image_b64 = message.get("image_base64")
        if not isinstance(image_b64, str):
            return {"ok": False, "request_id": request_id, "error": "image_base64 is required"}
        try:
            image_bytes = base64.b64decode(image_b64, validate=True)
        except Exception as exc:
            return {"ok": False, "request_id": request_id, "error": f"invalid image_base64: {exc}"}

        started = time.perf_counter()
        result = await asyncio.to_thread(self.detectors.detect, image_bytes, targets, threshold)
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
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recording System r4 proc2: MediaPipe object detection over AF_UNIX")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    return parser.parse_args(argv)


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    app_config = AppConfig.from_file(args.config)
    server = Proc2Server(app_config.proc2)
    try:
        await server.serve_forever()
    finally:
        await server.close()
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
