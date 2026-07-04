from __future__ import annotations

from datetime import datetime, timezone
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
from myrecorder.hls import parse_m3u8_text
from myrecorder.recording import HlsRecordingTask


def make_config(tmp_path: Path) -> AppConfig:
    source_hls_dir = tmp_path / "source-hls"
    source_hls_dir.mkdir()
    return AppConfig(
        base_dir=tmp_path,
        paths=PathsConfig(
            source_hls_dir=source_hls_dir,
            recordings_dir=tmp_path / "recordings",
            frame_storage_dir=tmp_path / "frames",
        ),
        rtsp=RtspConfig(url="rtsp://camera.local/stream"),
        hls=HlsConfig(),
        frames=FrameConfig(ta_seconds=5.0),
        ai=AiConfig(),
        recording=RecordingConfig(),
        slack=SlackConfig(),
    )


def test_initial_playlist_starts_at_ta_seconds_window(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    playlist_text = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:40
#EXT-X-PROGRAM-DATE-TIME:2026-06-21T01:02:00+00:00
#EXTINF:2.000,
segment_0000000040.ts
#EXTINF:2.000,
segment_0000000041.ts
#EXTINF:2.000,
segment_0000000042.ts
#EXTINF:2.000,
segment_0000000043.ts
#EXT-X-ENDLIST
"""
    playlist = parse_m3u8_text(playlist_text, config.source_playlist_path)
    frame_timestamp = datetime(2026, 6, 21, 1, 2, 7, tzinfo=timezone.utc)
    start_time = frame_timestamp.replace(second=2)
    task = HlsRecordingTask(config, manager=object(), recording_id=1, start_time=start_time)

    initial_segments = task._initial_segments(playlist)
    output = task._initial_playlist_text(playlist, initial_segments)

    assert [segment.sequence for segment in initial_segments] == [41, 42, 43]
    assert "#EXT-X-MEDIA-SEQUENCE:41" in output
    assert "segment_0000000040.ts" not in output
    assert "segment_0000000041.ts" in output
    assert "segment_0000000043.ts" in output
    assert "#EXT-X-ENDLIST" not in output
