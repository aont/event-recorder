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
