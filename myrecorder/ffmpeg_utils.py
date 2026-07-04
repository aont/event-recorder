from __future__ import annotations

import asyncio
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Awaitable, Callable, TextIO

from .config import AppConfig

LineCallback = Callable[[str], Awaitable[None] | None]

_OPENING_RE = re.compile(r"Opening ['\"](?P<path>[^'\"]+)['\"]")


def log_stamp() -> str:
    # datetime.astimezone() with no argument uses the host process' local timezone.
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def utc_stamp() -> str:
    # Backward-compatible alias for callers that imported the old helper name.
    return log_stamp()


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
            print(f"{log_stamp()} {process_name} {stream_name}: {text}", file=output, flush=True)
        if line_callback is not None:
            result = line_callback(text)
            if asyncio.iscoroutine(result):
                await result


@dataclass(slots=True)
class SegmentLogEvent:
    path: Path
    line: str


def clean_source_hls_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"{log_stamp()} app: cleaned source HLS dir {path}", file=sys.stderr, flush=True)


def _segment_number_regex(segment_pattern: str) -> re.Pattern[str] | None:
    match = re.search(r"%(?:0\d+)?d", segment_pattern)
    if match is None:
        return None
    prefix = re.escape(segment_pattern[: match.start()])
    suffix = re.escape(segment_pattern[match.end() :])
    return re.compile(f"^{prefix}(?P<number>\\d+){suffix}$")


def next_hls_start_number(source_hls_dir: Path, segment_pattern: str) -> int:
    pattern = _segment_number_regex(segment_pattern)
    if pattern is None or not source_hls_dir.exists():
        return 0

    max_number: int | None = None
    for child in source_hls_dir.iterdir():
        if not child.is_file():
            continue
        match = pattern.match(child.name)
        if match is None:
            continue
        number = int(match.group("number"))
        max_number = number if max_number is None else max(max_number, number)
    if max_number is None:
        return 0
    return max_number + 1


class FfmpegHlsTask:
    def __init__(self, config: AppConfig, segment_events: asyncio.Queue[SegmentLogEvent]) -> None:
        self.config = config
        self.segment_events = segment_events
        self._proc: asyncio.subprocess.Process | None = None
        self._seen_log_paths: set[Path] = set()
        self._stopping = False
        self._stop_event = asyncio.Event()

    def _next_start_number(self) -> int:
        return next_hls_start_number(self.config.paths.source_hls_dir, self.config.hls.segment_pattern)

    def command(self) -> list[str]:
        cfg = self.config
        start_number = self._next_start_number()
        cmd: list[str] = [
            cfg.hls.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            cfg.hls.loglevel,
        ]
        cmd.extend(cfg.input.ffmpeg_argv)
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
                "-start_number",
                str(start_number),
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
        while not self._stopping:
            returncode = await self._run_once()
            if self._stopping:
                return returncode
            sleep_seconds = self.config.hls.restart_sleep_seconds
            print(
                f"{log_stamp()} app: ffmpeg HLS exited with code {returncode}; "
                f"restarting in {sleep_seconds:g}s",
                file=sys.stderr,
                flush=True,
            )
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=sleep_seconds)
            except asyncio.TimeoutError:
                pass
            # Keep the existing source HLS files on restart. The next ffmpeg
            # launch uses a start number beyond the existing segment files and
            # append_list so the playlist can continue without overwriting or
            # hiding previously generated segments.
        return 0

    async def _run_once(self) -> int:
        cmd = self.command()
        print(f"{log_stamp()} app: starting ffmpeg HLS: {' '.join(cmd)}", file=sys.stderr, flush=True)
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
            print(f"{log_stamp()} app: ffmpeg HLS exited with code {returncode}", file=sys.stderr, flush=True)
            return returncode
        finally:
            for task in (stdout_task, stderr_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
            self._proc = None

    async def terminate(self) -> None:
        self._stopping = True
        self._stop_event.set()
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
    print(f"{log_stamp()} {process_name}: starting: {' '.join(map(str, cmd))}", file=sys.stderr, flush=True)
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
        print(f"{log_stamp()} {process_name}: exited with code {returncode}", file=sys.stderr, flush=True)
        return returncode
    finally:
        for task in (stdout_task, stderr_task):
            if not task.done():
                task.cancel()
