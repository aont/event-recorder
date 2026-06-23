from __future__ import annotations

import asyncio
import base64
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from .ipc import read_json_message, write_json_message


class AiClient:
    def __init__(
        self,
        *,
        socket_path: Path,
        targets: list[str],
        score_threshold: float,
        timeout_seconds: float,
        max_message_bytes: int,
    ) -> None:
        self.socket_path = socket_path
        self.targets = targets
        self.score_threshold = score_threshold
        self.timeout_seconds = timeout_seconds
        self.max_message_bytes = max_message_bytes

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
        request = {
            "type": "analyze_frame",
            "request_id": str(uuid.uuid4()),
            "frame_id": frame_id,
            "timestamp": timestamp.isoformat(),
            "segment_sequence": segment_sequence,
            "segment_uri": segment_uri,
            "offset_seconds": offset_seconds,
            "targets": self.targets,
            "score_threshold": self.score_threshold,
            "image_format": "jpeg",
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        }
        async def _roundtrip() -> dict[str, Any]:
            reader, writer = await asyncio.open_unix_connection(str(self.socket_path))
            try:
                await write_json_message(writer, request)
                response = await read_json_message(reader, max_bytes=self.max_message_bytes)
                if response is None:
                    raise RuntimeError("AI server closed the connection without a response")
                return response
            finally:
                writer.close()
                await writer.wait_closed()

        return await asyncio.wait_for(_roundtrip(), timeout=self.timeout_seconds)
