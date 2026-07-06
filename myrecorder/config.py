from __future__ import annotations

import math
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _section(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"[{name}] must be a TOML table")
    return value


def _list(value: Any, default: list[str] | None = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return list(value)
    raise TypeError(f"expected a string list, got {value!r}")


def _expand(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))


def _path(value: str | Path, base_dir: Path) -> Path:
    p = Path(_expand(str(value)))
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


@dataclass(frozen=True)
class PathsConfig:
    source_hls_dir: Path = Path("source-hls")
    recordings_dir: Path = Path("recordings")
    frame_storage_dir: Path = Path("frames")

    @classmethod
    def from_toml(cls, data: Mapping[str, Any], base_dir: Path) -> "PathsConfig":
        return cls(
            source_hls_dir=_path(data.get("source_hls_dir", cls.source_hls_dir), base_dir),
            recordings_dir=_path(data.get("recordings_dir", cls.recordings_dir), base_dir),
            frame_storage_dir=_path(data.get("frame_storage_dir", cls.frame_storage_dir), base_dir),
        )


@dataclass(frozen=True)
class InputConfig:
    ffmpeg_argv: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "InputConfig":
        return cls(ffmpeg_argv=[_expand(arg) for arg in _list(data.get("ffmpeg_argv"))])


@dataclass(frozen=True)
class HlsConfig:
    ffmpeg_bin: str = "ffmpeg"
    playlist_name: str = "live.m3u8"
    segment_pattern: str = "segment_%010d.ts"
    segment_seconds: float = 2.0
    retain_segments: int = 27
    delete_threshold: int = 2
    start_number_source: str = "generic"
    hls_flags: list[str] = field(default_factory=lambda: ["delete_segments", "program_date_time", "temp_file"])
    loglevel: str = "info"
    echo_ffmpeg_logs: bool = False
    stream_args: list[str] = field(default_factory=lambda: ["-map", "0:v:0", "-map", "0:a?", "-c", "copy"])
    output_args: list[str] = field(default_factory=list)
    clean_source_on_start: bool = True
    restart_sleep_seconds: float = 5.0

    @classmethod
    def from_toml(
        cls, data: Mapping[str, Any], frames: "FrameConfig" | None = None, ai: "AiConfig" | None = None
    ) -> "HlsConfig":
        default = cls()
        frames = frames or FrameConfig()
        ai = ai or AiConfig()
        segment_seconds = default.segment_seconds
        if frames.tc_seconds <= 0:
            raise ValueError("[frames].tc_seconds must be > 0")
        frames_per_segment = max(1, math.ceil(segment_seconds / frames.tc_seconds))
        analysis_window_seconds = frames_per_segment * ai.timeout_seconds
        recording_window_seconds = frames.ta_seconds + frames.tb_seconds
        retain_segments = max(1, math.ceil((analysis_window_seconds + recording_window_seconds) / segment_seconds) + 2)
        return cls(
            ffmpeg_bin=str(data.get("ffmpeg_bin", default.ffmpeg_bin)),
            retain_segments=retain_segments,
            restart_sleep_seconds=float(data.get("restart_sleep_seconds", default.restart_sleep_seconds)),
        )

    @property
    def hls_flags_arg(self) -> str:
        flags = list(dict.fromkeys([*self.hls_flags, "append_list"]))
        return "+".join(flags)


@dataclass(frozen=True)
class FrameConfig:
    tc_seconds: float = 1.0
    ta_seconds: float = 10.0
    tb_seconds: float = 10.0
    jpeg_quality: int = 3
    save_frames: bool = False
    delete_frames_for_deleted_segments: bool = True
    skip_existing_segments_on_start: bool = False
    echo_m3u8_progress_logs: bool = False

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "FrameConfig":
        default = cls()
        return cls(
            tc_seconds=float(data.get("tc_seconds", default.tc_seconds)),
            ta_seconds=float(data.get("ta_seconds", default.ta_seconds)),
            tb_seconds=float(data.get("tb_seconds", default.tb_seconds)),
            jpeg_quality=int(data.get("jpeg_quality", default.jpeg_quality)),
            save_frames=bool(data.get("save_frames", default.save_frames)),
            delete_frames_for_deleted_segments=bool(
                data.get("delete_frames_for_deleted_segments", default.delete_frames_for_deleted_segments)
            ),
            skip_existing_segments_on_start=bool(
                data.get("skip_existing_segments_on_start", default.skip_existing_segments_on_start)
            ),
        )


@dataclass(frozen=True)
class AiConfig:
    server_url: str = "http://127.0.0.1:8080"
    unix_socket_path: Path | None = None
    target_objects: list[str] = field(default_factory=lambda: ["cat"])
    score_threshold: float = 0.4
    timeout_seconds: float = 20.0
    max_results: int = -1
    min_confirming_frames: int = 2
    confirmation_window_seconds: float = 3.0

    @classmethod
    def from_toml(cls, data: Mapping[str, Any], base_dir: Path) -> "AiConfig":
        default = cls()
        socket_value = data.get("unix_socket_path")
        return cls(
            server_url=str(data.get("server_url", default.server_url)).rstrip("/"),
            unix_socket_path=_path(socket_value, base_dir) if socket_value else None,
            target_objects=_list(data.get("target_objects"), default.target_objects),
            score_threshold=float(data.get("score_threshold", default.score_threshold)),
            timeout_seconds=float(data.get("timeout_seconds", default.timeout_seconds)),
            max_results=int(data.get("max_results", default.max_results)),
            min_confirming_frames=int(data.get("min_confirming_frames", default.min_confirming_frames)),
            confirmation_window_seconds=float(
                data.get("confirmation_window_seconds", default.confirmation_window_seconds)
            ),
        )


@dataclass(frozen=True)
class RecordingConfig:
    poll_interval_seconds: float = 0.5
    convert_to_mp4: bool = True
    mp4_filename: str = "recording.mp4"
    mp4_args: list[str] = field(default_factory=lambda: ["-c", "copy", "-movflags", "+faststart"])
    use_local_time_for_filenames: bool = True
    delete_dir_after_slack_upload: bool = True

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "RecordingConfig":
        default = cls()
        return cls(
            poll_interval_seconds=float(data.get("poll_interval_seconds", default.poll_interval_seconds)),
            convert_to_mp4=bool(data.get("convert_to_mp4", default.convert_to_mp4)),
            mp4_filename=str(data.get("mp4_filename", default.mp4_filename)),
            mp4_args=_list(data.get("mp4_args"), default.mp4_args),
            use_local_time_for_filenames=bool(
                data.get("use_local_time_for_filenames", default.use_local_time_for_filenames)
            ),
            delete_dir_after_slack_upload=bool(
                data.get("delete_dir_after_slack_upload", default.delete_dir_after_slack_upload)
            ),
        )


@dataclass(frozen=True)
class SlackConfig:
    enabled: bool = False
    bot_token: str | None = None
    channel_id: str | None = None
    title_prefix: str = "Recording"
    initial_comment: str = ""

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "SlackConfig":
        default = cls()
        token = data.get("bot_token")
        channel = data.get("channel_id")
        return cls(
            enabled=bool(data.get("enabled", default.enabled)),
            bot_token=str(token) if token else None,
            channel_id=str(channel) if channel else None,
            title_prefix=str(data.get("title_prefix", default.title_prefix)),
            initial_comment=str(data.get("initial_comment", default.initial_comment)),
        )


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    paths: PathsConfig
    input: InputConfig
    hls: HlsConfig
    frames: FrameConfig
    ai: AiConfig
    recording: RecordingConfig
    slack: SlackConfig

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        base_dir = config_path.parent
        paths = PathsConfig.from_toml(_section(data, "paths"), base_dir)
        ai_section = _section(data, "ai")
        frames = FrameConfig.from_toml(_section(data, "frames"))
        ai = AiConfig.from_toml(ai_section, base_dir)
        return cls(
            base_dir=base_dir,
            paths=paths,
            input=InputConfig.from_toml(_section(data, "input")),
            hls=HlsConfig.from_toml(_section(data, "hls"), frames, ai),
            frames=frames,
            ai=ai,
            recording=RecordingConfig(),
            slack=SlackConfig.from_toml(_section(data, "slack")),
        )

    def ensure_directories(self) -> None:
        self.paths.source_hls_dir.mkdir(parents=True, exist_ok=True)
        self.paths.recordings_dir.mkdir(parents=True, exist_ok=True)
        if self.frames.save_frames:
            self.paths.frame_storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def source_playlist_path(self) -> Path:
        return self.paths.source_hls_dir / self.hls.playlist_name

    @property
    def source_segment_pattern_path(self) -> Path:
        return self.paths.source_hls_dir / self.hls.segment_pattern

    def validate_app(self) -> None:
        if not self.input.ffmpeg_argv:
            raise ValueError("[input].ffmpeg_argv must include ffmpeg input arguments")
        if "-i" not in self.input.ffmpeg_argv:
            raise ValueError("[input].ffmpeg_argv must include an ffmpeg -i input")
        if any(arg.startswith("rtsp://example") for arg in self.input.ffmpeg_argv):
            raise ValueError("[input].ffmpeg_argv must be set to a real input URL")
        if self.frames.tc_seconds <= 0:
            raise ValueError("[frames].tc_seconds must be > 0")
        if self.frames.ta_seconds < 0:
            raise ValueError("[frames].ta_seconds must be >= 0")
        if self.frames.tb_seconds < 0:
            raise ValueError("[frames].tb_seconds must be >= 0")
        if self.hls.retain_segments <= 0:
            raise ValueError("[hls].retain_segments must be > 0")
        if self.hls.restart_sleep_seconds < 0:
            raise ValueError("[hls].restart_sleep_seconds must be >= 0")
        if not self.ai.server_url:
            raise ValueError("[ai].server_url must not be empty")
        if self.ai.min_confirming_frames <= 0:
            raise ValueError("[ai].min_confirming_frames must be > 0")
        if self.ai.confirmation_window_seconds <= 0:
            raise ValueError("[ai].confirmation_window_seconds must be > 0")
