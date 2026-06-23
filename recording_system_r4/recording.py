from __future__ import annotations

import asyncio
import errno
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from .config import AppConfig
from .ffmpeg_utils import run_ffmpeg_with_prefixed_logs, utc_stamp
from .hls import HlsPlaylist, HlsSegment, append_lines, load_m3u8, strip_endlist
from .slack_upload import upload_file_to_slack


def _safe_timestamp(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _segment_destination_relative_path(segment: HlsSegment) -> Path:
    parsed = urlparse(segment.uri)
    if parsed.scheme == "file":
        p = Path(unquote(parsed.path))
        return Path(p.name) if p.is_absolute() else p
    raw_path = segment.uri.split("?", 1)[0]
    p = Path(unquote(raw_path))
    return Path(p.name) if p.is_absolute() else p


def _hardlink_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError as exc:
        if exc.errno in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            print(
                f"{utc_stamp()} recording: hardlink failed ({exc}); falling back to copy2 for {src}",
                file=sys.stderr,
                flush=True,
            )
            shutil.copy2(src, dst)
        else:
            raise


@dataclass(slots=True)
class RecordingState:
    recording_id: int
    destination_dir: Path
    playlist_path: Path
    copied_sequences: set[int]
    last_copied_end_time: datetime | None = None


class RecordingManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._lock = asyncio.Lock()
        self._end_time: datetime | None = None
        self._current_task: asyncio.Task[None] | None = None
        self._current_recording_id: int | None = None
        self._next_recording_id = 1
        self._all_tasks: set[asyncio.Task[None]] = set()

    async def on_target_detected(self, frame_timestamp: datetime, analysis: dict[str, Any]) -> None:
        if frame_timestamp.tzinfo is None:
            frame_timestamp = frame_timestamp.replace(tzinfo=timezone.utc)
        requested_end = frame_timestamp + timedelta(seconds=self.config.frames.tb_seconds)
        async with self._lock:
            if self._end_time is None or requested_end > self._end_time:
                self._end_time = requested_end
                print(
                    f"{utc_stamp()} recording: end time updated to {self._end_time.isoformat()} "
                    f"from frame {analysis.get('frame_id')}",
                    file=sys.stderr,
                    flush=True,
                )

            if self._current_task is None or self._current_task.done():
                recording_id = self._next_recording_id
                self._next_recording_id += 1
                self._current_recording_id = recording_id
                task = asyncio.create_task(self._run_recording(recording_id), name=f"recording-{recording_id}")
                self._current_task = task
                self._all_tasks.add(task)
                task.add_done_callback(self._all_tasks.discard)
                print(f"{utc_stamp()} recording: started task {recording_id}", file=sys.stderr, flush=True)

    async def get_end_time(self) -> datetime | None:
        async with self._lock:
            return self._end_time

    async def mark_capture_closed(self, recording_id: int) -> None:
        async with self._lock:
            if self._current_recording_id == recording_id:
                self._current_task = None
                self._current_recording_id = None
                self._end_time = None
                print(f"{utc_stamp()} recording: capture {recording_id} closed", file=sys.stderr, flush=True)

    async def _run_recording(self, recording_id: int) -> None:
        task = HlsRecordingTask(self.config, self, recording_id)
        try:
            await task.run()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"{utc_stamp()} recording: task {recording_id} failed: {exc!r}", file=sys.stderr, flush=True)
            await self.mark_capture_closed(recording_id)

    async def close(self) -> None:
        tasks = list(self._all_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


class HlsRecordingTask:
    def __init__(self, config: AppConfig, manager: RecordingManager, recording_id: int) -> None:
        self.config = config
        self.manager = manager
        self.recording_id = recording_id

    async def run(self) -> None:
        playlist = await self._wait_for_source_playlist()
        state = self._create_destination(playlist)
        await self._capture_loop(state)
        mp4_path = await self._convert_to_mp4(state)
        if mp4_path is not None:
            title = f"{self.config.slack.title_prefix} {state.destination_dir.name}"
            await upload_file_to_slack(self.config.slack, mp4_path, title=title)

    async def _wait_for_source_playlist(self) -> HlsPlaylist:
        while True:
            playlist = load_m3u8(self.config.source_playlist_path)
            if playlist is not None and playlist.segments:
                return playlist
            await asyncio.sleep(self.config.recording.poll_interval_seconds)

    def _create_destination(self, playlist: HlsPlaylist) -> RecordingState:
        now = datetime.now(timezone.utc)
        destination_dir = self.config.paths.recordings_dir / f"rec_{_safe_timestamp(now)}_{self.recording_id:04d}"
        destination_dir.mkdir(parents=True, exist_ok=False)
        destination_playlist = destination_dir / self.config.hls.playlist_name

        copied_sequences: set[int] = set()
        last_end: datetime | None = None
        for segment in playlist.segments:
            try:
                self._copy_segment(segment, destination_dir)
                copied_sequences.add(segment.sequence)
                if segment.end_time is not None:
                    last_end = segment.end_time
            except FileNotFoundError:
                print(
                    f"{utc_stamp()} recording: source segment vanished before copy: seq={segment.sequence} uri={segment.uri}",
                    file=sys.stderr,
                    flush=True,
                )

        destination_playlist.write_text(strip_endlist(playlist.raw_text), encoding="utf-8")
        print(
            f"{utc_stamp()} recording: copied initial playlist with {len(copied_sequences)} segments to {destination_dir}",
            file=sys.stderr,
            flush=True,
        )
        return RecordingState(
            recording_id=self.recording_id,
            destination_dir=destination_dir,
            playlist_path=destination_playlist,
            copied_sequences=copied_sequences,
            last_copied_end_time=last_end,
        )

    def _copy_segment(self, segment: HlsSegment, destination_dir: Path) -> None:
        src = segment.resolved_path(self.config.source_playlist_path)
        if not src.exists():
            raise FileNotFoundError(src)
        dst = destination_dir / _segment_destination_relative_path(segment)
        _hardlink_or_copy(src, dst)

    async def _capture_loop(self, state: RecordingState) -> None:
        try:
            while True:
                playlist = load_m3u8(self.config.source_playlist_path)
                if playlist is not None:
                    self._copy_and_append_new_segments(state, playlist)

                end_time = await self.manager.get_end_time()
                if self._has_reached_end_time(state, end_time):
                    append_lines(state.playlist_path, ["#EXT-X-ENDLIST"])
                    await self.manager.mark_capture_closed(self.recording_id)
                    return

                await asyncio.sleep(self.config.recording.poll_interval_seconds)
        except Exception:
            await self.manager.mark_capture_closed(self.recording_id)
            raise

    def _copy_and_append_new_segments(self, state: RecordingState, playlist: HlsPlaylist) -> None:
        new_segments = [segment for segment in playlist.segments if segment.sequence not in state.copied_sequences]
        if not new_segments:
            return

        if state.copied_sequences:
            expected_next = max(state.copied_sequences) + 1
            if new_segments[0].sequence > expected_next:
                print(
                    f"{utc_stamp()} recording: warning: source playlist skipped from seq {expected_next} "
                    f"to {new_segments[0].sequence}; retention may be too small",
                    file=sys.stderr,
                    flush=True,
                )

        append_batch: list[str] = []
        appended_count = 0
        for segment in new_segments:
            try:
                self._copy_segment(segment, state.destination_dir)
            except FileNotFoundError:
                print(
                    f"{utc_stamp()} recording: warning: could not copy new segment seq={segment.sequence} uri={segment.uri}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            state.copied_sequences.add(segment.sequence)
            append_batch.extend(segment.raw_entry_lines)
            appended_count += 1
            if segment.end_time is not None:
                state.last_copied_end_time = segment.end_time

        if append_batch:
            append_lines(state.playlist_path, append_batch)
            print(
                f"{utc_stamp()} recording: appended {appended_count} source segments to {state.playlist_path}",
                file=sys.stderr,
                flush=True,
            )

    def _has_reached_end_time(self, state: RecordingState, end_time: datetime | None) -> bool:
        if end_time is None:
            return False
        if state.last_copied_end_time is not None:
            return state.last_copied_end_time >= end_time
        return datetime.now(timezone.utc) >= end_time

    async def _convert_to_mp4(self, state: RecordingState) -> Path | None:
        if not self.config.recording.convert_to_mp4:
            print(
                f"{utc_stamp()} recording: MP4 conversion disabled for {state.destination_dir}",
                file=sys.stderr,
                flush=True,
            )
            return None
        output_path = state.destination_dir / self.config.recording.mp4_filename
        cmd = [
            self.config.hls.ffmpeg_bin,
            "-hide_banner",
            "-nostdin",
            "-y",
            "-loglevel",
            "info",
            "-allowed_extensions",
            "ALL",
            "-protocol_whitelist",
            "file,crypto,data",
            "-i",
            str(state.playlist_path),
            *self.config.recording.mp4_args,
            str(output_path),
        ]
        rc = await run_ffmpeg_with_prefixed_logs(cmd, process_name="ffmpeg-convert", cwd=state.destination_dir)
        if rc != 0:
            raise RuntimeError(f"MP4 conversion failed with ffmpeg exit code {rc}")
        return output_path
