from pathlib import Path

from myrecorder.config import PathsConfig


def test_paths_are_resolved_directly_against_config_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()

    paths = PathsConfig.from_toml(
        {
            "source_hls_dir": "source-hls",
            "recordings_dir": "recordings",
            "frame_storage_dir": "frames",
        },
        config_dir,
    )

    assert paths.source_hls_dir == config_dir / "source-hls"
    assert paths.recordings_dir == config_dir / "recordings"
    assert paths.frame_storage_dir == config_dir / "frames"


def test_paths_config_has_no_work_dir() -> None:
    assert "work_dir" not in PathsConfig.__dataclass_fields__


def test_ai_config_has_no_socket_path() -> None:
    from myrecorder.config import AiConfig

    assert "socket_path" not in AiConfig.__dataclass_fields__


def test_app_config_uses_defaults_when_optional_sections_are_omitted(tmp_path: Path) -> None:
    from myrecorder.config import AppConfig

    config_path = tmp_path / "config.toml"
    config_path.write_text("[rtsp]\nurl = 'rtsp://camera.local/stream'\n")

    config = AppConfig.from_file(config_path)

    assert config.paths.source_hls_dir == tmp_path / "source-hls"
    assert config.paths.recordings_dir == tmp_path / "recordings"
    assert config.paths.frame_storage_dir == tmp_path / "frames"
    assert config.hls.ffmpeg_bin == "ffmpeg"
    assert config.hls.restart_sleep_seconds == 5.0
    assert config.ai.model_path == tmp_path / "models" / "efficientdet_lite0.tflite"
    assert config.slack.initial_comment == ""


def test_hls_retain_segments_is_calculated_from_runtime_settings() -> None:
    from myrecorder.config import AiConfig, FrameConfig, HlsConfig

    hls = HlsConfig.from_toml(
        {},
        frames=FrameConfig(tc_seconds=1.0, tb_seconds=10.0),
        ai=AiConfig(timeout_seconds=20.0),
    )

    assert hls.segment_seconds == 2.0
    assert hls.retain_segments == 27
