from pathlib import Path

from myrecorder.hls import parse_m3u8_text


def test_parse_playlist_media_sequence_and_program_date_time():
    text = """#EXTM3U
#EXT-X-VERSION:3
#EXT-X-TARGETDURATION:2
#EXT-X-MEDIA-SEQUENCE:42
#EXT-X-PROGRAM-DATE-TIME:2026-06-21T01:02:03.000Z
#EXTINF:2.000,
segment_0000000042.ts
#EXT-X-PROGRAM-DATE-TIME:2026-06-21T01:02:05.000Z
#EXTINF:2.000,
segment_0000000043.ts
"""
    playlist = parse_m3u8_text(text, Path("/tmp/live.m3u8"))
    assert playlist.media_sequence == 42
    assert playlist.target_duration == 2
    assert len(playlist.segments) == 2
    assert playlist.segments[0].sequence == 42
    assert playlist.segments[1].sequence == 43
    assert playlist.segments[0].program_date_time.isoformat() == "2026-06-21T01:02:03+00:00"
