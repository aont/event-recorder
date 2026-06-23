from __future__ import annotations

import asyncio
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, TextIO, Sequence

from .config import AppConfig

LineCallback = Callable[[str], Awaitable[None] | None]

_OPENING_RE = re.compile(r"Opening ['\"](?P<path>[^'\"]+)['\"]")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


async def pump_stream(
    stream: asyncio.StreamReader | None,
    *,
    process_name: str,
    stream_name: str,
    output: TextIO | None,
    line_callback: LineCallback | None = None,
) -> None:
    if stream is None:
        return
    while True:
        raw = await stream.readline()
        if not raw:
            return
        text = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if output is not None:
            print(f"{utc_stamp()} {process_name} {stream_name}: {text}", file=output, flush=True)
        if line_callback is not None:
            result = line_callback(text)
            if asyncio.iscoroutine(result):
                await result


@dataclass(slots=True)
class SegmentLogEvent:
    path: Path
    line: str


class FfmpegHlsTask:
    def __init__(self, config: AppConfig, segment_events: asyncio.Queue[SegmentLogEvent]) -> None:
        self.config = config
        self.segment_events = segment_events
        self._proc: asyncio.subprocess.Process | None = None
        self._seen_log_paths: set[Path] = set()

    def command(self) -> list[str]:
        cfg = self.config
        cmd: list[str] = [
            cfg.hls.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            cfg.hls.loglevel,
        ]
        cmd.extend(cfg.rtsp.input_args)
        if cfg.rtsp.transport:
            cmd.extend(["-rtsp_transport", cfg.rtsp.transport])
        cmd.extend(["-i", cfg.rtsp.url])
        cmd.extend(cfg.hls.stream_args)
        cmd.extend(
            [
                "-f",
                "hls",
                "-hls_time",
                str(cfg.hls.segment_seconds),
                "-hls_list_size",
                str(cfg.hls.retain_segments),
                "-hls_delete_threshold",
                str(cfg.hls.delete_threshold),
                "-hls_start_number_source",
                cfg.hls.start_number_source,
                "-hls_flags",
                cfg.hls.hls_flags_arg,
                "-hls_segment_filename",
                cfg.hls.segment_pattern,
            ]
        )
        cmd.extend(cfg.hls.output_args)
        cmd.append(cfg.hls.playlist_name)
        return cmd

    async def _handle_log_line(self, line: str) -> None:
        for match in _OPENING_RE.finditer(line):
            raw_path = match.group("path")
            p = Path(raw_path)
            # FFmpeg also opens the playlist and temporary playlist files. Only
            # segment-like files are useful as a wake-up signal.
            if p.suffix == ".tmp" and p.name.endswith((".ts.tmp", ".m4s.tmp", ".mp4.tmp", ".aac.tmp", ".vtt.tmp")):
                p = p.with_name(p.name.removesuffix(".tmp"))
            elif p.suffix not in {".ts", ".m4s", ".mp4", ".aac", ".vtt"}:
                continue
            resolved = p if p.is_absolute() else (self.config.paths.source_hls_dir / p).resolve()
            if resolved in self._seen_log_paths:
                continue
            self._seen_log_paths.add(resolved)
            await self.segment_events.put(SegmentLogEvent(path=resolved, line=line))

    async def run(self) -> int:
        cmd = self.command()
        print(f"{utc_stamp()} proc1: starting ffmpeg HLS: {' '.join(cmd)}", file=sys.stderr, flush=True)
        self._proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.config.paths.source_hls_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_task = asyncio.create_task(
            pump_stream(
                self._proc.stdout,
                process_name="ffmpeg",
                stream_name="stdout",
                output=sys.stdout if self.config.hls.echo_ffmpeg_logs else None,
                line_callback=self._handle_log_line,
            )
        )
        stderr_task = asyncio.create_task(
            pump_stream(
                self._proc.stderr,
                process_name="ffmpeg",
                stream_name="stderr",
                output=sys.stderr if self.config.hls.echo_ffmpeg_logs else None,
                line_callback=self._handle_log_line,
            )
        )
        try:
            returncode = await self._proc.wait()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            print(f"{utc_stamp()} proc1: ffmpeg HLS exited with code {returncode}", file=sys.stderr, flush=True)
            return returncode
        finally:
            for task in (stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            await self.terminate()

    async def terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()


def split_mjpeg_stream(data: bytes) -> list[bytes]:
    frames: list[bytes] = []
    pos = 0
    while True:
        start = data.find(b"\xff\xd8", pos)
        if start < 0:
            break
        end = data.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        end += 2
        frames.append(data[start:end])
        pos = end
    return frames


async def extract_jpeg_frames(
    *,
    ffmpeg_bin: str,
    segment_path: Path,
    every_seconds: float,
    jpeg_quality: int,
) -> list[bytes]:
    fps = 1.0 / every_seconds
    vf = f"fps=fps={fps:.8f}"
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(segment_path),
        "-vf",
        vf,
        "-an",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        str(jpeg_quality),
        "pipe:1",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    if proc.returncode != 0:
        message = stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg frame extraction failed for {segment_path}: {message}")
    return split_mjpeg_stream(stdout)


async def run_ffmpeg_with_prefixed_logs(
    cmd: Sequence[str],
    *,
    process_name: str,
    cwd: Path | None = None,
    echo_logs: bool = False,
) -> int:
    print(f"{utc_stamp()} {process_name}: starting: {' '.join(map(str, cmd))}", file=sys.stderr, flush=True)
    proc = await asyncio.create_subprocess_exec(
        *map(str, cmd),
        cwd=str(cwd) if cwd else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout_task = asyncio.create_task(
        pump_stream(proc.stdout, process_name=process_name, stream_name="stdout", output=sys.stdout if echo_logs else None)
    )
    stderr_task = asyncio.create_task(
        pump_stream(proc.stderr, process_name=process_name, stream_name="stderr", output=sys.stderr if echo_logs else None)
    )
    try:
        returncode = await proc.wait()
        await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        print(f"{utc_stamp()} {process_name}: exited with code {returncode}", file=sys.stderr, flush=True)
        return returncode
    finally:
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
