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
from myrecorder.ffmpeg_utils import FfmpegHlsTask, SegmentLogEvent, next_hls_start_number


def make_config(source_hls_dir):
    return AppConfig(
        base_dir=source_hls_dir.parent,
        paths=PathsConfig(
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


def test_next_hls_start_number_uses_next_existing_segment_number(tmp_path):
    source_hls_dir = tmp_path / "source-hls"
    source_hls_dir.mkdir()
    (source_hls_dir / "segment_0000000003.ts").write_text("old segment")
    (source_hls_dir / "segment_0000000011.ts").write_text("newest segment")
    (source_hls_dir / "segment_0000000011.ts.tmp").write_text("ignored temp segment")
    (source_hls_dir / "live.m3u8").write_text("ignored playlist")

    assert next_hls_start_number(source_hls_dir, "segment_%010d.ts") == 12


def test_ffmpeg_command_appends_to_existing_hls_with_non_overlapping_start_number(tmp_path):
    source_hls_dir = tmp_path / "source-hls"
    source_hls_dir.mkdir()
    (source_hls_dir / "segment_0000000041.ts").write_text("old segment")

    task = FfmpegHlsTask(make_config(source_hls_dir), asyncio.Queue[SegmentLogEvent]())

    cmd = task.command()
    assert cmd[cmd.index("-start_number") + 1] == "42"
    assert "append_list" in cmd[cmd.index("-hls_flags") + 1].split("+")


def test_ffmpeg_restart_preserves_source_hls_dir_and_seen_paths(tmp_path):
    source_hls_dir = tmp_path / "source-hls"
    source_hls_dir.mkdir()
    playlist = source_hls_dir / "live.m3u8"
    playlist.write_text("existing playlist")
    nested_dir = source_hls_dir / "nested"
    nested_dir.mkdir()
    stale_segment = nested_dir / "stale.ts"
    stale_segment.write_text("existing segment")

    class RestartingTask(FfmpegHlsTask):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.runs = 0

        async def _run_once(self) -> int:
            self.runs += 1
            if self.runs == 1:
                self._seen_log_paths.add(playlist.resolve())
                return 1
            assert playlist.read_text() == "existing playlist"
            assert stale_segment.read_text() == "existing segment"
            assert self._seen_log_paths == {playlist.resolve()}
            self._stopping = True
            return 0

    task = RestartingTask(make_config(source_hls_dir), asyncio.Queue[SegmentLogEvent]())

    assert asyncio.run(task.run()) == 0
    assert task.runs == 2
