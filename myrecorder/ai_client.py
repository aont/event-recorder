from __future__ import annotations

import asyncio
import functools
import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from .ai_worker import analyze_frame_in_worker, init_ai_worker


class AiClient:
    """Async app-side AI facade backed by ProcessPoolExecutor workers."""

    def __init__(
        self,
        *,
        model_path: Path,
        targets: list[str],
        score_threshold: float,
        timeout_seconds: float,
        max_results: int,
        workers: int,
    ) -> None:
        self.model_path = model_path
        self.targets = targets
        self.score_threshold = score_threshold
        self.timeout_seconds = timeout_seconds
        self.max_results = max_results
        self.workers = workers
        self._executor: ProcessPoolExecutor | None = None

    def start(self) -> None:
        if self._executor is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"model_path does not exist: {self.model_path}")
        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=init_ai_worker,
            initargs=(str(self.model_path), self.max_results),
        )

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
        self.start()
        assert self._executor is not None
        loop = asyncio.get_running_loop()
        request_id = str(uuid.uuid4())
        call = functools.partial(
            analyze_frame_in_worker,
            image_bytes,
            self.targets,
            self.score_threshold,
        )
        result = await asyncio.wait_for(
            loop.run_in_executor(self._executor, call),
            timeout=self.timeout_seconds,
        )
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
            **result,
        }

    async def close(self) -> None:
        executor = self._executor
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
