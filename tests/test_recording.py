from __future__ import annotations

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
from myrecorder.recording import HlsRecordingTask, RecordingManager, RecordingState


def make_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        base_dir=tmp_path,
        paths=PathsConfig(
            work_dir=tmp_path,
            source_hls_dir=tmp_path / "source-hls",
            recordings_dir=tmp_path / "recordings",
            frame_storage_dir=tmp_path / "frames",
        ),
        rtsp=RtspConfig(url="rtsp://camera.local/stream"),
        hls=HlsConfig(),
        frames=FrameConfig(),
        ai=AiConfig(model_path=tmp_path / "model.tflite"),
        recording=RecordingConfig(),
        slack=SlackConfig(),
    )


def test_source_sequence_rewind_closes_recording(tmp_path, capsys):
    config = make_config(tmp_path)
    task = HlsRecordingTask(config, RecordingManager(config), recording_id=3)
    state = RecordingState(
        recording_id=3,
        destination_dir=tmp_path / "recordings" / "rec",
        playlist_path=tmp_path / "recordings" / "rec" / "live.m3u8",
        copied_sequences={98, 99, 100},
    )
    playlist = parse_m3u8_text(
        "\n".join(
            [
                "#EXTM3U",
                "#EXT-X-TARGETDURATION:2",
                "#EXT-X-MEDIA-SEQUENCE:0",
                "#EXTINF:2.000000,",
                "segment_000000000000.ts",
                "#EXTINF:2.000000,",
                "segment_000000000001.ts",
            ]
        ),
        config.source_playlist_path,
    )

    assert task._source_sequence_rewound(state, playlist)
    assert "source playlist sequence rewound" in capsys.readouterr().err
