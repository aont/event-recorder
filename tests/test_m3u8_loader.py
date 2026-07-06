from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from myrecorder.config import (
    AiConfig,
    AppConfig,
    FrameConfig,
    HlsConfig,
    PathsConfig,
    InputConfig,
    RecordingConfig,
    SlackConfig,
)
from myrecorder.ffmpeg_utils import SegmentLogEvent
from myrecorder.m3u8_loader import ExtractedFrame, M3u8LoadingTask


def make_config(base_dir: Path) -> AppConfig:
    source_hls_dir = base_dir / "source-hls"
    source_hls_dir.mkdir()
    return AppConfig(
        base_dir=base_dir,
        paths=PathsConfig(
            source_hls_dir=source_hls_dir,
            recordings_dir=base_dir / "recordings",
            frame_storage_dir=base_dir / "frames",
        ),
        input=InputConfig(ffmpeg_argv=["-i", "rtsp://camera.local/stream"]),
        hls=HlsConfig(),
        frames=FrameConfig(),
        ai=AiConfig(),
        recording=RecordingConfig(poll_interval_seconds=0.01),
        slack=SlackConfig(),
    )


class CountingM3u8LoadingTask(M3u8LoadingTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.process_count = 0
        self.processed = asyncio.Event()

    async def process_once(self) -> None:
        self.process_count += 1
        self.processed.set()


def test_m3u8_loader_waits_for_ffmpeg_segment_log_event_instead_of_polling(tmp_path):
    async def scenario() -> None:
        segment_events: asyncio.Queue[SegmentLogEvent] = asyncio.Queue()
        task = CountingM3u8LoadingTask(
            config=make_config(tmp_path),
            segment_events=segment_events,
            ai_client=object(),
            recording_manager=object(),
        )
        runner = asyncio.create_task(task.run())
        try:
            await asyncio.sleep(0.05)
            assert task.process_count == 0

            await segment_events.put(SegmentLogEvent(path=tmp_path / "segment_0000000001.ts", line="Opening segment"))
            await asyncio.wait_for(task.processed.wait(), timeout=1)
            assert task.process_count == 1
        finally:
            runner.cancel()
            await asyncio.gather(runner, return_exceptions=True)

    asyncio.run(scenario())


class FakeAiClient:
    def __init__(self, analyses: list[dict[str, Any]]) -> None:
        self.analyses = list(analyses)

    async def analyze_frame(self, **kwargs: Any) -> dict[str, Any]:
        return self.analyses.pop(0)


class FakeRecordingManager:
    def __init__(self) -> None:
        self.detections = []

    async def on_target_detected(self, frame_timestamp: datetime, analysis: dict[str, Any]) -> None:
        self.detections.append((frame_timestamp, analysis))


def make_frame(frame_id: str, timestamp: datetime) -> ExtractedFrame:
    return ExtractedFrame(
        frame_id=frame_id,
        segment_sequence=1,
        segment_uri="segment_0000000001.ts",
        segment_duration=2.0,
        offset_seconds=0.0,
        timestamp=timestamp,
        image_bytes=b"jpeg",
    )


def target_analysis(label: str = "cat") -> dict[str, Any]:
    return {"ok": True, "has_target": True, "detections": [{"label": label, "score": 0.9, "bbox": {}}]}


def test_single_detection_does_not_start_recording(tmp_path):
    async def scenario() -> None:
        recorder = FakeRecordingManager()
        task = M3u8LoadingTask(
            config=make_config(tmp_path),
            segment_events=asyncio.Queue(),
            ai_client=FakeAiClient([target_analysis()]),
            recording_manager=recorder,
        )

        await task._analyze_frame(make_frame("frame-1", datetime(2026, 1, 1, tzinfo=timezone.utc)))

        assert recorder.detections == []

    asyncio.run(scenario())


def test_consecutive_detections_for_same_label_start_recording(tmp_path):
    async def scenario() -> None:
        recorder = FakeRecordingManager()
        task = M3u8LoadingTask(
            config=make_config(tmp_path),
            segment_events=asyncio.Queue(),
            ai_client=FakeAiClient([target_analysis(), target_analysis()]),
            recording_manager=recorder,
        )
        first = datetime(2026, 1, 1, tzinfo=timezone.utc)
        second = first + timedelta(seconds=1)

        await task._analyze_frame(make_frame("frame-1", first))
        await task._analyze_frame(make_frame("frame-2", second))

        assert len(recorder.detections) == 1
        assert recorder.detections[0][0] == second

    asyncio.run(scenario())
