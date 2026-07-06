from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import AppConfig
from .ffmpeg_utils import SegmentLogEvent, extract_jpeg_frames, log_stamp
from .hls import HlsPlaylist, HlsSegment, load_m3u8
from .recording import RecordingManager

if TYPE_CHECKING:
    from .ai_client import AiClient


@dataclass(slots=True)
class ExtractedFrame:
    frame_id: str
    segment_sequence: int
    segment_uri: str
    segment_duration: float
    offset_seconds: float
    timestamp: datetime
    image_bytes: bytes
    storage_path: Path | None = None
    analysis: dict[str, Any] | None = None


class M3u8LoadingTask:
    def __init__(
        self,
        *,
        config: AppConfig,
        segment_events: asyncio.Queue[SegmentLogEvent],
        ai_client: AiClient,
        recording_manager: RecordingManager,
    ) -> None:
        self.config = config
        self.segment_events = segment_events
        self.ai_client = ai_client
        self.recording_manager = recording_manager
        self.last_processed_sequence: int | None = None
        self.frames_by_segment: dict[int, list[ExtractedFrame]] = {}
        self.positive_frame_history: dict[str, list[datetime]] = defaultdict(list)

    def _progress(self, message: str) -> None:
        if self.config.frames.echo_m3u8_progress_logs:
            print(f"{log_stamp()} m3u8-loader: {message}", file=sys.stderr, flush=True)

    async def run(self) -> None:
        while True:
            event = await self.segment_events.get()
            self._progress(f"segment log event: {event.path}")
            # Give FFmpeg a short interval to finish atomic playlist rename.
            await asyncio.sleep(0.15)
            await self.process_once()

    async def process_once(self) -> None:
        playlist = load_m3u8(self.config.source_playlist_path)
        if playlist is None or not playlist.segments:
            return

        self._delete_frames_for_removed_segments(playlist)

        if self.last_processed_sequence is None:
            if self.config.frames.skip_existing_segments_on_start:
                self.last_processed_sequence = playlist.last_sequence
                self._progress(f"skipping existing source segments through seq {self.last_processed_sequence}")
                return
            self.last_processed_sequence = playlist.media_sequence - 1

        if playlist.media_sequence > self.last_processed_sequence + 1:
            print(
                f"{log_stamp()} m3u8-loader: warning: source playlist advanced from seq "
                f"{self.last_processed_sequence + 1} to {playlist.media_sequence}; unprocessed segments were lost",
                file=sys.stderr,
                flush=True,
            )
            self.last_processed_sequence = playlist.media_sequence - 1

        unprocessed = [s for s in playlist.segments if s.sequence > self.last_processed_sequence]
        if len(unprocessed) >= 2:
            print(
                f"{log_stamp()} m3u8-loader: warning: {len(unprocessed)} unprocessed segments are queued; "
                "consider increasing retain_segments or reducing AI latency",
                file=sys.stderr,
                flush=True,
            )

        for segment in unprocessed:
            await self._process_segment(segment)
            self.last_processed_sequence = max(self.last_processed_sequence, segment.sequence)

    def _delete_frames_for_removed_segments(self, playlist: HlsPlaylist) -> None:
        if not self.config.frames.delete_frames_for_deleted_segments:
            return
        active_sequences = playlist.active_sequences()
        removed = [seq for seq in self.frames_by_segment if seq not in active_sequences]
        for seq in removed:
            frames = self.frames_by_segment.pop(seq)
            for frame in frames:
                if frame.storage_path is not None:
                    try:
                        frame.storage_path.unlink()
                    except FileNotFoundError:
                        pass
            if self.config.frames.save_frames:
                seq_dir = self.config.paths.frame_storage_dir / str(seq)
                try:
                    seq_dir.rmdir()
                except OSError:
                    pass
            self._progress(f"deleted frames for removed segment seq={seq}")

    async def _process_segment(self, segment: HlsSegment) -> None:
        try:
            segment_path = segment.resolved_path(self.config.source_playlist_path)
        except ValueError as exc:
            print(f"{log_stamp()} m3u8-loader: {exc}", file=sys.stderr, flush=True)
            return

        if not segment_path.exists():
            print(
                f"{log_stamp()} m3u8-loader: warning: segment file is missing: seq={segment.sequence} {segment_path}",
                file=sys.stderr,
                flush=True,
            )
            return

        self._progress(
            f"extracting frames from seq={segment.sequence} "
            f"duration={segment.duration:.3f}s uri={segment.uri}"
        )
        try:
            images = await extract_jpeg_frames(
                ffmpeg_bin=self.config.hls.ffmpeg_bin,
                segment_path=segment_path,
                every_seconds=self.config.frames.tc_seconds,
                jpeg_quality=self.config.frames.jpeg_quality,
            )
        except Exception as exc:
            print(
                f"{log_stamp()} m3u8-loader: frame extraction failed for seq={segment.sequence}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return

        base_time = segment.program_date_time or datetime.now(timezone.utc)
        frames: list[ExtractedFrame] = []
        for index, image_bytes in enumerate(images):
            offset = index * self.config.frames.tc_seconds
            timestamp = base_time + timedelta(seconds=offset)
            frame = ExtractedFrame(
                frame_id=f"seq{segment.sequence}-offset{int(offset * 1000):09d}",
                segment_sequence=segment.sequence,
                segment_uri=segment.uri,
                segment_duration=segment.duration,
                offset_seconds=offset,
                timestamp=timestamp,
                image_bytes=image_bytes,
            )
            if self.config.frames.save_frames:
                frame.storage_path = self._save_frame(frame)
            frames.append(frame)

        self.frames_by_segment[segment.sequence] = frames
        self._progress(f"extracted {len(frames)} frames from seq={segment.sequence}")

        for frame in frames:
            await self._analyze_frame(frame)

    def _save_frame(self, frame: ExtractedFrame) -> Path:
        seq_dir = self.config.paths.frame_storage_dir / str(frame.segment_sequence)
        seq_dir.mkdir(parents=True, exist_ok=True)
        path = seq_dir / f"{frame.frame_id}.jpg"
        path.write_bytes(frame.image_bytes)
        return path

    async def _analyze_frame(self, frame: ExtractedFrame) -> None:
        try:
            analysis = await self.ai_client.analyze_frame(
                frame_id=frame.frame_id,
                timestamp=frame.timestamp,
                image_bytes=frame.image_bytes,
                segment_sequence=frame.segment_sequence,
                segment_uri=frame.segment_uri,
                offset_seconds=frame.offset_seconds,
            )
        except Exception as exc:
            print(
                f"{log_stamp()} m3u8-loader: AI request failed for frame={frame.frame_id}: {exc!r}",
                file=sys.stderr,
                flush=True,
            )
            return

        frame.analysis = analysis
        if not analysis.get("ok"):
            print(
                f"{log_stamp()} m3u8-loader: AI error for frame={frame.frame_id}: {analysis.get('error')}",
                file=sys.stderr,
                flush=True,
            )
            return

        if analysis.get("has_target"):
            print(
                f"{log_stamp()} m3u8-loader: target detected frame={frame.frame_id} "
                f"detections={analysis.get('detections')}",
                file=sys.stderr,
                flush=True,
            )
            if self._has_confirmed_target(frame.timestamp, analysis):
                await self.recording_manager.on_target_detected(frame.timestamp, analysis)

    def _has_confirmed_target(self, timestamp: datetime, analysis: dict[str, Any]) -> bool:
        window_start = timestamp - timedelta(seconds=self.config.ai.confirmation_window_seconds)
        confirmed = False
        labels = {
            str(detection.get("label"))
            for detection in analysis.get("detections") or []
            if detection.get("label")
        }
        for label in labels:
            history = [seen_at for seen_at in self.positive_frame_history[label] if seen_at >= window_start]
            history.append(timestamp)
            self.positive_frame_history[label] = history
            if len(history) >= self.config.ai.min_confirming_frames:
                confirmed = True

        # Keep unrelated labels bounded even if they are not present in this frame.
        for label in list(self.positive_frame_history):
            self.positive_frame_history[label] = [
                seen_at for seen_at in self.positive_frame_history[label] if seen_at >= window_start
            ]
            if not self.positive_frame_history[label]:
                del self.positive_frame_history[label]

        return confirmed
