from __future__ import annotations

import asyncio

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
from myrecorder.ffmpeg_utils import FfmpegHlsTask, SegmentLogEvent


def make_config(source_hls_dir):
    return AppConfig(
        base_dir=source_hls_dir.parent,
        paths=PathsConfig(
            work_dir=source_hls_dir.parent,
            source_hls_dir=source_hls_dir,
            recordings_dir=source_hls_dir.parent / "recordings",
            frame_storage_dir=source_hls_dir.parent / "frames",
        ),
        rtsp=RtspConfig(url="rtsp://camera.local/stream"),
        hls=HlsConfig(restart_sleep_seconds=0),
        frames=FrameConfig(),
        ai=AiConfig(model_path=source_hls_dir.parent / "model.tflite"),
        recording=RecordingConfig(),
        slack=SlackConfig(),
    )


def test_ffmpeg_restart_cleans_source_hls_dir(tmp_path):
    source_hls_dir = tmp_path / "source-hls"
    source_hls_dir.mkdir()
    (source_hls_dir / "live.m3u8").write_text("stale playlist")
    nested_dir = source_hls_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "stale.ts").write_text("stale segment")

    class RestartingTask(FfmpegHlsTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.runs = 0

        async def _run_once(self) -> int:
            self.runs += 1
            if self.runs == 1:
                self._seen_log_paths.add((source_hls_dir / "live.m3u8").resolve())
                return 1
            assert list(source_hls_dir.iterdir()) == []
            assert self._seen_log_paths == set()
            self._stopping = True
            return 0

    task = RestartingTask(make_config(source_hls_dir), asyncio.Queue[SegmentLogEvent]())

    assert asyncio.run(task.run()) == 0
    assert task.runs == 2
