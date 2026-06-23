from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.parse import unquote, urlparse

_EXTINF_RE = re.compile(r"^#EXTINF:([0-9]+(?:\.[0-9]+)?)(?:,(.*))?$")
_SEGMENT_TAG_PREFIXES = (
    "#EXT-X-DISCONTINUITY",
    "#EXT-X-BYTERANGE",
    "#EXT-X-MAP",
    "#EXT-X-KEY",
    "#EXT-X-GAP",
)


def parse_hls_datetime(value: str) -> datetime | None:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


@dataclass(slots=True)
class HlsSegment:
    sequence: int
    uri: str
    duration: float
    title: str = ""
    program_date_time: datetime | None = None
    raw_entry_lines: list[str] = field(default_factory=list)

    def resolved_path(self, playlist_path: Path) -> Path:
        parsed = urlparse(self.uri)
        if parsed.scheme and parsed.scheme != "file":
            raise ValueError(f"remote segment URI is not supported for hard-linking: {self.uri}")
        raw_path = parsed.path if parsed.scheme == "file" else self.uri.split("?", 1)[0]
        p = Path(unquote(raw_path))
        return p if p.is_absolute() else (playlist_path.parent / p).resolve()

    @property
    def end_time(self) -> datetime | None:
        if self.program_date_time is None:
            return None
        return self.program_date_time + timedelta(seconds=self.duration)


@dataclass(slots=True)
class HlsPlaylist:
    path: Path
    raw_text: str
    lines: list[str]
    media_sequence: int = 0
    target_duration: int | None = None
    version: int | None = None
    segments: list[HlsSegment] = field(default_factory=list)

    @property
    def last_sequence(self) -> int | None:
        if not self.segments:
            return None
        return self.segments[-1].sequence

    def active_sequences(self) -> set[int]:
        return {segment.sequence for segment in self.segments}

    def by_sequence(self) -> dict[int, HlsSegment]:
        return {segment.sequence: segment for segment in self.segments}

    def header_without_endlist(self) -> list[str]:
        header: list[str] = []
        first_segment_uri = self.segments[0].uri if self.segments else None
        for line in self.lines:
            if line == "#EXT-X-ENDLIST":
                continue
            if first_segment_uri is not None and line == first_segment_uri:
                break
            # If no segments exist, this returns the whole playlist minus ENDLIST.
            header.append(line)
        return header


def parse_m3u8_text(text: str, path: Path) -> HlsPlaylist:
    # Preserve ordering while normalizing CRLF. Blank lines are irrelevant for the
    # HLS tags used here, so they are kept only if they are in raw segment entries.
    lines = [line.rstrip("\r") for line in text.splitlines()]
    playlist = HlsPlaylist(path=path, raw_text=text, lines=lines)

    media_sequence = 0
    pending_lines: list[str] = []
    pending_duration: float | None = None
    pending_title = ""
    pending_pdt: datetime | None = None

    for line in lines:
        if line.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                media_sequence = int(line.split(":", 1)[1].strip())
                playlist.media_sequence = media_sequence
            except ValueError:
                pass
            continue

        if line.startswith("#EXT-X-TARGETDURATION:"):
            try:
                playlist.target_duration = int(float(line.split(":", 1)[1].strip()))
            except ValueError:
                pass
            continue

        if line.startswith("#EXT-X-VERSION:"):
            try:
                playlist.version = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
            continue

        if line.startswith("#EXT-X-PROGRAM-DATE-TIME:"):
            pending_lines.append(line)
            pending_pdt = parse_hls_datetime(line.split(":", 1)[1])
            continue

        extinf = _EXTINF_RE.match(line)
        if extinf:
            pending_lines.append(line)
            pending_duration = float(extinf.group(1))
            pending_title = extinf.group(2) or ""
            continue

        if line.startswith("#") or line == "":
            # Tags immediately before EXTINF/URI belong to the next segment.
            # This covers EXT-X-DISCONTINUITY and similar per-segment tags.
            if pending_lines or line.startswith(_SEGMENT_TAG_PREFIXES):
                pending_lines.append(line)
            continue

        # URI line.
        sequence = media_sequence + len(playlist.segments)
        duration = pending_duration if pending_duration is not None else 0.0
        raw_entry_lines = [*pending_lines, line]
        playlist.segments.append(
            HlsSegment(
                sequence=sequence,
                uri=line,
                duration=duration,
                title=pending_title,
                program_date_time=pending_pdt,
                raw_entry_lines=raw_entry_lines,
            )
        )
        pending_lines = []
        pending_duration = None
        pending_title = ""
        pending_pdt = None

    _fill_missing_program_date_times(playlist.segments)
    return playlist


def _fill_missing_program_date_times(segments: Iterable[HlsSegment]) -> None:
    cursor: datetime | None = None
    for segment in segments:
        if segment.program_date_time is None and cursor is not None:
            segment.program_date_time = cursor
        if segment.program_date_time is not None:
            cursor = segment.program_date_time + timedelta(seconds=segment.duration)


def load_m3u8(path: Path) -> HlsPlaylist | None:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None
    if "#EXTM3U" not in text[:128]:
        return None
    return parse_m3u8_text(text, path)


def strip_endlist(text: str) -> str:
    lines = [line for line in text.splitlines() if line.rstrip("\r") != "#EXT-X-ENDLIST"]
    return "\n".join(lines).rstrip("\n") + "\n"


def append_lines(path: Path, lines: Iterable[str]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line)
            if not line.endswith("\n"):
                f.write("\n")
