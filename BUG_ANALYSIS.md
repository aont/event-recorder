# The reason the video uploaded to Slack is not continuous and jumps at around the 10-second mark.

## Conclusion

The root cause is **not the Slack upload process, but the append logic for the recording HLS playlist**.

In `recording.py`, `_copy_and_append_new_segments()` treats **all segments that have not yet been copied as newly arrived segments**. As a result, older segments that were intentionally excluded at the start of recording because of `ta_seconds` are appended to the end of the recording playlist during subsequent iterations.

Consequently, the playlist used as input for MP4 conversion is no longer in chronological order, causing the MP4 uploaded to Slack to jump back to earlier footage partway through playback.

---

## Problem Area

When recording begins, the recording start time is set to `frame_timestamp - [frames].ta_seconds`, as described in the README. Likewise, recording ends at `frame_timestamp + [frames].tb_seconds`. ([GitHub][1])

By default:

* `ta_seconds = 10.0`
* `tb_seconds = 10.0`

([GitHub][2])

Therefore, the observed behavior—"jumping around the 10-second mark"—matches the default `ta_seconds = 10`.

The core issue is the following code:

```python
new_segments = [
    segment
    for segment in playlist.segments
    if segment.sequence not in state.copied_sequences
]
```

This logic simply assumes that **any segment that has not been copied is a new segment**. ([GitHub][3])

However, during recording initialization, `_initial_segments()` only copies segments whose `segment.end_time > self.start_time`. In other words, segments older than the `ta_seconds` recording window are **intentionally excluded** from the initial recording playlist. ([GitHub][3])

Since those older segments are never added to `state.copied_sequences`, the next call to `_copy_and_append_new_segments()` incorrectly treats them as newly arrived segments and appends them to the end of the recording playlist. ([GitHub][3])

---

## Typical Failure Scenario

Assume 2-second HLS segments and a recording window that starts 10 seconds before the detection time.

### Initial Recording

Source playlist:

```text
segment_0000000080.ts
segment_0000000081.ts
...
segment_0000000095.ts
segment_0000000096.ts
...
segment_0000000103.ts
```

Because `ta_seconds = 10`, the recording playlist initially copies only:

```text
segment_0000000096.ts
segment_0000000097.ts
...
segment_0000000103.ts
```

This behavior is correct.

### Immediately Afterward

The source playlist still contains segments `80–95`.

Since these segments were intentionally excluded from the initial copy, they are not present in `state.copied_sequences`. The current implementation therefore considers them "new" and appends them to the recording playlist.

The resulting playlist becomes:

```text
segment_0000000096.ts
segment_0000000097.ts
...
segment_0000000103.ts
segment_0000000080.ts   <-- playback jumps backward here
segment_0000000081.ts
...
segment_0000000095.ts
segment_0000000104.ts
```

Because this playlist is later converted to MP4 using FFmpeg, the MP4 uploaded to Slack also jumps backward during playback. The Slack integration itself merely uploads the completed MP4 using `files_upload_v2`; it is not responsible for the playback issue. ([GitHub][3])

---

## Proposed Fix

Instead of appending every segment that has not yet been copied, `_copy_and_append_new_segments()` should append **only segments whose sequence number is greater than the last sequence already present in the recording playlist**.

### Suggested Implementation

Replace `_copy_and_append_new_segments()` in `myrecorder/recording.py` with the following implementation:

```python
def _copy_and_append_new_segments(self, state: RecordingState, playlist: HlsPlaylist) -> None:
    if state.copied_sequences:
        expected_next = max(state.copied_sequences) + 1

        # Important:
        # Do not append older retained source segments that were intentionally
        # excluded by the ta_seconds start window.
        new_segments = [
            segment
            for segment in playlist.segments
            if segment.sequence >= expected_next
        ]

        if new_segments and new_segments[0].sequence > expected_next:
            print(
                f"{log_stamp()} recording: warning: source playlist skipped from seq {expected_next} "
                f"to {new_segments[0].sequence}; retention may be too small",
                file=sys.stderr,
                flush=True,
            )
    else:
        new_segments = list(playlist.segments)

    if not new_segments:
        return

    append_batch: list[str] = []
    appended_count = 0

    for segment in new_segments:
        try:
            self._copy_segment(segment, state.destination_dir)
        except FileNotFoundError:
            print(
                f"{log_stamp()} recording: warning: could not copy new segment seq={segment.sequence} uri={segment.uri}",
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
            f"{log_stamp()} recording: appended {appended_count} source segments to {state.playlist_path}",
            file=sys.stderr,
            flush=True,
        )
```

The critical change is:

```python
if segment.sequence >= expected_next
```

This prevents segments intentionally excluded at the beginning of the recording from being appended later.

---

## How to Verify

By default, the recording directory is deleted after a successful Slack upload. The README explains that setting `delete_dir_after_slack_upload = false` preserves the recording directory. ([GitHub][1])

First, add the following to `config.toml`:

```toml
[recording]
delete_dir_after_slack_upload = false
```

Then inspect the generated playlist:

```bash
grep 'segment_.*\.ts' recordings/rec_*/live.m3u8
```

If the playlist is affected by this bug, the sequence numbers will jump backward:

```text
segment_0000000096.ts
segment_0000000097.ts
...
segment_0000000103.ts
segment_0000000080.ts
segment_0000000081.ts
```

After applying the fix, the sequence numbers should increase monotonically:

```text
segment_0000000096.ts
segment_0000000097.ts
...
segment_0000000103.ts
segment_0000000104.ts
segment_0000000105.ts
```

---

## Additional Note

`_copy_and_append_new_segments()` already contains a warning that reports:

> "source playlist skipped ... retention may be too small"

However, with the current implementation, older retained segments appear first in `new_segments`, preventing this warning from triggering in some cases. ([GitHub][3])

Therefore, the observed behavior is more accurately attributed to **incorrect filtering of segments to append**, rather than insufficient HLS retention.

[1]: https://github.com/aont/event-recorder "GitHub - aont/event-recorder: When an event is detected, it captures video retroactively from before the detection time, convert clips to MP4, and optionally upload them to Slack. · GitHub"
[2]: https://github.com/aont/event-recorder/blob/main/myrecorder/config.py "event-recorder/myrecorder/config.py at main · aont/event-recorder · GitHub"
[3]: https://github.com/aont/event-recorder/blob/main/myrecorder/recording.py "event-recorder/myrecorder/recording.py at main · aont/event-recorder · GitHub"
