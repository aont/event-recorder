from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import aiohttp


class AiClient:
    """Async detector facade backed by a detection-server HTTP API."""

    def __init__(
        self,
        *,
        server_url: str,
        unix_socket_path: Path | None,
        targets: list[str],
        score_threshold: float,
        timeout_seconds: float,
        max_results: int,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.unix_socket_path = unix_socket_path
        self.targets = targets
        self.score_threshold = score_threshold
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self._session: aiohttp.ClientSession | None = None

    def start(self) -> None:
        # The aiohttp session is created lazily in the running event loop.
        return

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            connector: aiohttp.BaseConnector | None = None
            if self.unix_socket_path is not None:
                connector = aiohttp.UnixConnector(path=str(self.unix_socket_path))
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
        return self._session

    def _detect_url(self) -> str:
        if self.unix_socket_path is not None:
            return "http://localhost/v1/detect"
        return f"{self.server_url}/v1/detect"

    def _request_config(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "score_threshold": self.score_threshold,
            "max_results": self.max_results,
        }
        if self.targets:
            options["category_allowlist"] = self.targets
        return {"object_detector_options": options}

    @staticmethod
    def _bbox_to_dict(bbox: dict[str, Any]) -> dict[str, int]:
        return {
            "x": int(bbox.get("origin_x", 0)),
            "y": int(bbox.get("origin_y", 0)),
            "width": int(bbox.get("width", 0)),
            "height": int(bbox.get("height", 0)),
        }

    def _normalize_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        target_set = set(self.targets)
        detections: list[dict[str, Any]] = []
        result = payload.get("result") or {}
        for detection in result.get("detections") or []:
            bbox = self._bbox_to_dict(detection.get("bounding_box") or {})
            for category in detection.get("categories") or []:
                name = category.get("category_name") or category.get("display_name") or ""
                score = float(category.get("score") or 0.0)
                if score < self.score_threshold:
                    continue
                if target_set and name not in target_set:
                    continue
                detections.append({"label": name, "score": score, "bbox": bbox})
        return {
            "has_target": len(detections) > 0,
            "num_targets": len(detections),
            "detections": detections,
        }

    async def analyze_frame(
        self,
        *,
        frame_id: str,
        timestamp: datetime,
        image_bytes: bytes,
        segment_sequence: int,
        segment_uri: str,
        offset_seconds: float,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        form = aiohttp.FormData()
        form.add_field("config", json.dumps(self._request_config()), content_type="application/json")
        form.add_field("image", image_bytes, filename=f"{frame_id}.jpg", content_type="image/jpeg")

        session = await self._get_session()
        started = time.perf_counter()
        async with session.post(self._detect_url(), data=form) as response:
            response.raise_for_status()
            payload = await response.json()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        normalized = self._normalize_response(payload)
        return {
            "ok": True,
            "request_id": request_id,
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "segment_sequence": segment_sequence,
            "segment_uri": segment_uri,
            "offset_seconds": offset_seconds,
            "targets": self.targets,
            "score_threshold": self.score_threshold,
            "processing_ms": elapsed_ms,
            **normalized,
        }

    async def close(self) -> None:
        session = self._session
        self._session = None
        if session is not None:
            await session.close()
