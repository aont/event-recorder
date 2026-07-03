from __future__ import annotations

import asyncio
from pathlib import Path

from myrecorder.config import (
    AiConfig,
    AppConfig,
    FrameConfig,
    HlsConfig,
    PathsConfig,
    RecordingConfig,
    RtspConfig,
    SlackConfig,
)
from myrecorder.ffmpeg_utils import SegmentLogEvent
from myrecorder.m3u8_loader import M3u8LoadingTask


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
        rtsp=RtspConfig(url="rtsp://camera.local/stream"),
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
