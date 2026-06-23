from __future__ import annotations

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
    work_dir: Path = Path("./var/recording-r4")
    source_hls_dir: Path = Path("source-hls")
    recordings_dir: Path = Path("recordings")
    frame_storage_dir: Path = Path("frames")

    @classmethod
    def from_toml(cls, data: Mapping[str, Any], base_dir: Path) -> "PathsConfig":
        raw_work = data.get("work_dir", cls.work_dir)
        work_dir = _path(raw_work, base_dir)

        def child(name: str, default: Path) -> Path:
            raw = data.get(name, default)
            p = Path(_expand(str(raw)))
            return p if p.is_absolute() else (work_dir / p).resolve()

        return cls(
            work_dir=work_dir,
            source_hls_dir=child("source_hls_dir", cls.source_hls_dir),
            recordings_dir=child("recordings_dir", cls.recordings_dir),
            frame_storage_dir=child("frame_storage_dir", cls.frame_storage_dir),
        )


@dataclass(frozen=True)
class RtspConfig:
    url: str
    transport: str = "tcp"
    input_args: list[str] = field(default_factory=list)

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "RtspConfig":
        url = str(data.get("url", "")).strip()
        return cls(
            url=_expand(url),
            transport=str(data.get("transport", "tcp")),
            input_args=_list(data.get("input_args")),
        )


@dataclass(frozen=True)
class HlsConfig:
    ffmpeg_bin: str = "ffmpeg"
    playlist_name: str = "live.m3u8"
    segment_pattern: str = "segment_%010d.ts"
    segment_seconds: float = 2.0
    retain_segments: int = 10
    delete_threshold: int = 2
    start_number_source: str = "generic"
    hls_flags: list[str] = field(default_factory=lambda: ["delete_segments", "program_date_time", "temp_file"])
    loglevel: str = "info"
    echo_ffmpeg_logs: bool = False
    stream_args: list[str] = field(default_factory=lambda: ["-map", "0:v:0", "-map", "0:a?", "-c", "copy"])
    output_args: list[str] = field(default_factory=list)
    clean_source_on_start: bool = True

    @classmethod
    def from_toml(cls, data: Mapping[str, Any]) -> "HlsConfig":
        default = cls()
        return cls(
            ffmpeg_bin=str(data.get("ffmpeg_bin", default.ffmpeg_bin)),
            playlist_name=str(data.get("playlist_name", default.playlist_name)),
            segment_pattern=str(data.get("segment_pattern", default.segment_pattern)),
            segment_seconds=float(data.get("segment_seconds", default.segment_seconds)),
            retain_segments=int(data.get("retain_segments", default.retain_segments)),
            delete_threshold=int(data.get("delete_threshold", default.delete_threshold)),
            start_number_source=str(data.get("start_number_source", default.start_number_source)),
            hls_flags=_list(data.get("hls_flags"), default.hls_flags),
            loglevel=str(data.get("loglevel", default.loglevel)),
            echo_ffmpeg_logs=bool(data.get("echo_ffmpeg_logs", default.echo_ffmpeg_logs)),
            stream_args=_list(data.get("stream_args"), default.stream_args),
            output_args=_list(data.get("output_args"), default.output_args),
            clean_source_on_start=bool(data.get("clean_source_on_start", default.clean_source_on_start)),
        )

    @property
    def hls_flags_arg(self) -> str:
        return "+".join(self.hls_flags)


@dataclass(frozen=True)
class FrameConfig:
    tc_seconds: float = 1.0
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
            tb_seconds=float(data.get("tb_seconds", default.tb_seconds)),
            jpeg_quality=int(data.get("jpeg_quality", default.jpeg_quality)),
            save_frames=bool(data.get("save_frames", default.save_frames)),
            delete_frames_for_deleted_segments=bool(
                data.get("delete_frames_for_deleted_segments", default.delete_frames_for_deleted_segments)
            ),
            skip_existing_segments_on_start=bool(
                data.get("skip_existing_segments_on_start", default.skip_existing_segments_on_start)
            ),
            echo_m3u8_progress_logs=bool(
                data.get("echo_m3u8_progress_logs", default.echo_m3u8_progress_logs)
            ),
        )


@dataclass(frozen=True)
class AiConfig:
    socket_path: Path = Path("/tmp/recording-r4-proc2.sock")
    target_objects: list[str] = field(default_factory=lambda: ["cat"])
    score_threshold: float = 0.4
    timeout_seconds: float = 20.0
    max_message_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_toml(cls, data: Mapping[str, Any], base_dir: Path) -> "AiConfig":
        default = cls()
        return cls(
            socket_path=_path(data.get("socket_path", default.socket_path), base_dir),
            target_objects=_list(data.get("target_objects"), default.target_objects),
            score_threshold=float(data.get("score_threshold", default.score_threshold)),
            timeout_seconds=float(data.get("timeout_seconds", default.timeout_seconds)),
            max_message_bytes=int(data.get("max_message_bytes", default.max_message_bytes)),
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
    initial_comment: str = "Recording uploaded."

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
class Proc2Config:
    socket_path: Path = Path("/tmp/recording-r4-proc2.sock")
    model_path: Path = Path("efficientdet_lite0.tflite")
    target_objects: list[str] = field(default_factory=lambda: ["cat"])
    score_threshold: float = 0.4
    max_results: int = -1
    max_message_bytes: int = 20 * 1024 * 1024

    @classmethod
    def from_toml(cls, data: Mapping[str, Any], ai: Mapping[str, Any], base_dir: Path) -> "Proc2Config":
        default = cls()
        model_raw = data.get("model_path", default.model_path)
        return cls(
            socket_path=_path(data.get("socket_path", ai.get("socket_path", default.socket_path)), base_dir),
            model_path=_path(model_raw, base_dir),
            target_objects=_list(data.get("target_objects", ai.get("target_objects")), default.target_objects),
            score_threshold=float(data.get("score_threshold", ai.get("score_threshold", default.score_threshold))),
            max_results=int(data.get("max_results", default.max_results)),
            max_message_bytes=int(data.get("max_message_bytes", ai.get("max_message_bytes", default.max_message_bytes))),
        )


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    paths: PathsConfig
    rtsp: RtspConfig
    hls: HlsConfig
    frames: FrameConfig
    ai: AiConfig
    recording: RecordingConfig
    slack: SlackConfig
    proc2: Proc2Config

    @classmethod
    def from_file(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as f:
            data = tomllib.load(f)
        base_dir = config_path.parent
        paths = PathsConfig.from_toml(_section(data, "paths"), base_dir)
        ai_section = _section(data, "ai")
        return cls(
            base_dir=base_dir,
            paths=paths,
            rtsp=RtspConfig.from_toml(_section(data, "rtsp")),
            hls=HlsConfig.from_toml(_section(data, "hls")),
            frames=FrameConfig.from_toml(_section(data, "frames")),
            ai=AiConfig.from_toml(ai_section, base_dir),
            recording=RecordingConfig.from_toml(_section(data, "recording")),
            slack=SlackConfig.from_toml(_section(data, "slack")),
            proc2=Proc2Config.from_toml(_section(data, "proc2"), ai_section, base_dir),
        )

    def ensure_directories(self) -> None:
        self.paths.work_dir.mkdir(parents=True, exist_ok=True)
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

    def validate_proc1(self) -> None:
        if not self.rtsp.url or self.rtsp.url.startswith("rtsp://example"):
            raise ValueError("[rtsp].url must be set to a real RTSP URL")
        if self.frames.tc_seconds <= 0:
            raise ValueError("[frames].tc_seconds must be > 0")
        if self.frames.tb_seconds < 0:
            raise ValueError("[frames].tb_seconds must be >= 0")
        if self.hls.retain_segments <= 0:
            raise ValueError("[hls].retain_segments must be > 0")
