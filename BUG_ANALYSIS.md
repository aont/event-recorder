# Bug Analysis

## Conclusion

The root cause is **not the Slack posting process itself, but a design flaw where the m3u8 loader stops processing new segments after the HLS-generating `ffmpeg` process restarts**.

The provided logs show `ffmpeg HLS exited with code 1` repeating every second, which means no HLS segments are being generated for recording at that point. Furthermore, in the current implementation of this repository, even if `ffmpeg` recovers, **the HLS sequence number resets to around 0, causing the m3u8 loader to incorrectly treat new segments as already processed and fail to resume frame analysis**.

I have not pushed any changes.

---

## Immediate Cause

### 1. The logs indicate HLS generation is failing, not Slack

The logs show the following:

```text
app: starting ffmpeg HLS: ffmpeg ... -i rtsp://127.0.0.1:8554/cam1 ...
app: ffmpeg HLS exited with code 1
app: ffmpeg HLS exited with code 1; restarting in 1s
app: cleaned source HLS dir ...
```

`FfmpegHlsTask.run()` restarts `ffmpeg` whenever it exits, deleting `source_hls_dir` before restarting. The relevant code calls `clean_source_hls_dir()` on abnormal termination and also clears `_seen_log_paths`. ([GitHub][1])

Slack uploads are only invoked after the recording task copies HLS segments and finishes MP4 conversion. `HlsRecordingTask.run()` executes in the order `_capture_loop()` → `_convert_to_mp4()` → `upload_file_to_slack()`. ([GitHub][2])

Therefore, if HLS segments are never generated, execution never reaches the Slack upload step.

---

## Most Likely Root Cause in the Code

### 2. `hls_start_number_source = "generic"` causes the sequence number to reset after restart

In the example configuration, `[hls] start_number_source = "generic"` is the default. ([GitHub][3])

According to the FFmpeg HLS muxer documentation, `hls_start_number_source=generic` uses `start_number`, whose default value is `0`. ([FFmpeg][4])

As a result, after `ffmpeg` restarts following long-running operation, both `#EXT-X-MEDIA-SEQUENCE` and filenames such as `segment_%012d.ts` restart from around 0.

Meanwhile, `M3u8LoadingTask` only considers segments whose sequence number is greater than `last_processed_sequence`:

```python
unprocessed = [s for s in playlist.segments if s.sequence > self.last_processed_sequence]
```

This implementation assumes sequence numbers increase monotonically. ([GitHub][5])

There is no logic to detect that the sequence has wrapped back to 0 after an `ffmpeg` restart and reset `last_processed_sequence`.

#### Example

Assuming `segment_seconds = 2.0`:

| Running Time | Approximate `last_processed_sequence` Before Restart | Time Until Processing Resumes After Restart |
| -----------: | ---------------------------------------------------: | ------------------------------------------: |
|     24 hours |                                              ~43,200 |                                   ~24 hours |
|       3 days |                                             ~129,600 |                                     ~3 days |
|       7 days |                                             ~302,400 |                                     ~7 days |

In other words, if HLS crashes once after long-term operation, AI detection stops for approximately the same duration even after recovery. Since AI detection stops, recording initiation and Slack posting also stop.

In the provided logs, `ffmpeg` is repeatedly exiting immediately with code 1, making the situation even worse: the sequence continually resets to 0 before it can increase. Under these conditions, posting effectively never resumes.

---

## Additional Issues Identified

### 3. The actual `ffmpeg` error is not being logged

The example configuration defaults `echo_ffmpeg_logs = false`, with comments stating that `ffmpeg` stdout/stderr are consumed internally but not printed. ([GitHub][3])

The implementation also simply reads stdout/stderr via `pump_stream()` without outputting them when `echo_ffmpeg_logs` is false. ([GitHub][1])

As a result, only `code 1` appears in the logs, while the actual underlying error could be something like:

```text
Connection refused
404 Not Found
Server returned 5XX
Invalid data found when processing input
Could not find codec parameters
Non-monotonous DTS
No space left on device
Permission denied
```

Since the input is `rtsp://127.0.0.1:8554/cam1`, the most likely causes are a stopped or hung local RTSP server/proxy, an incorrect stream name, or a camera-side disconnection.

---

## Immediate Mitigations

### A. Change `start_number_source` to `epoch_us`

Update the `[hls]` section in `config.toml`:

```toml
[hls]
start_number_source = "epoch_us"
```

`epoch_us` is supported by FFmpeg and uses the Unix epoch in microseconds as the start number. ([FFmpeg][4])

This ensures that after an `ffmpeg` restart, the sequence number remains well ahead of the previous `last_processed_sequence`, allowing the m3u8 loader to continue processing new segments.

`epoch` also improves the situation, but `epoch_us` is safer for avoiding collisions during rapid successive restarts.

### B. Enable `ffmpeg` logging for diagnosis

```toml
[hls]
echo_ffmpeg_logs = true
loglevel = "info"
```

The next time `ffmpeg HLS exited with code 1` occurs, the preceding RTSP connection error or codec/muxer error should become visible.

### C. Verify the RTSP input directly

Without using the application, run:

```bash
ffprobe -hide_banner -rtsp_transport tcp -i rtsp://127.0.0.1:8554/cam1
```

Or execute the same `ffmpeg` command shown in the application logs directly from the `source-hls` directory:

```bash
cd /home/naoto-aoki/Programs/20260621-cat-ai-002/tmp/var/recording-r4/source-hls

ffmpeg -hide_banner -nostdin -loglevel info \
  -rtsp_transport tcp \
  -i rtsp://127.0.0.1:8554/cam1 \
  -map 0:v:0 -map 0:a? -c copy \
  -f hls \
  -hls_time 2.0 \
  -hls_list_size 10 \
  -hls_delete_threshold 2 \
  -hls_start_number_source epoch_us \
  -hls_flags delete_segments+program_date_time+temp_file \
  -hls_segment_filename segment_%012d.ts \
  live.m3u8
```

If this reports `Connection refused`, the issue is with the RTSP server at `127.0.0.1:8554`. If it reports codec or muxer errors, the `stream_args` configuration should be reviewed.

---

## Proposed Permanent Fixes

### Fix 1: Detect sequence resets in the m3u8 loader

In `myrecorder/m3u8_loader.py`, `process_once()` should detect when the latest sequence number in the playlist becomes smaller than `last_processed_sequence`, indicating that the HLS source has been reset, and reset the loader state accordingly.

Example implementation:

```python
async def process_once(self) -> None:
    playlist = load_m3u8(self.config.source_playlist_path)
    if playlist is None or not playlist.segments:
        return

    # Handle FFmpeg restart with hls_start_number_source=generic,
    # where EXT-X-MEDIA-SEQUENCE moves backwards.
    if (
        self.last_processed_sequence is not None
        and playlist.last_sequence is not None
        and playlist.last_sequence < self.last_processed_sequence
    ):
        print(
            f"{utc_stamp()} m3u8-loader: source playlist sequence reset detected; "
            f"last_processed_sequence={self.last_processed_sequence}, "
            f"new_media_sequence={playlist.media_sequence}, "
            f"new_last_sequence={playlist.last_sequence}; resetting loader state",
            file=sys.stderr,
            flush=True,
        )
        self.last_processed_sequence = playlist.media_sequence - 1
        self.frames_by_segment.clear()

    self._delete_frames_for_removed_segments(playlist)

    ...
```

With this change, even if the sequence resets to 0 after an `ffmpeg` restart, the loader will resume processing the new playlist.

### Fix 2: Always log the tail of `ffmpeg` stderr on abnormal exit

Currently, if `echo_ffmpeg_logs=false`, no stderr output is visible. Even when logging is disabled, the last 20–50 lines of stderr should always be printed whenever `ffmpeg` exits with a non-zero status.

Suggested approach:

```python
# Store the stderr tail in deque(maxlen=50) within pump_stream().
# If returncode != 0, print the buffered tail to stderr.
```

This would immediately reveal the real cause behind future `code 1` failures.

### Fix 3: Add a capture loop timeout when HLS stops during recording

The recording task proceeds to MP4 conversion and Slack upload only after `_capture_loop()` exits. ([GitHub][2])

However, `_has_reached_end_time()` only checks whether `last_copied_end_time` has exceeded `end_time`. If HLS stops and no new segments arrive, `last_copied_end_time` never advances, potentially preventing the recording task from finishing. ([GitHub][2])

This can prevent even partial clips from being uploaded after detection. The recording task should either finalize a partial clip after a configurable timeout without new segments or abort cleanly and release the recording manager state.

---

## Recommended Priority

1. **Immediate operational mitigation**
   Change `start_number_source = "epoch_us"` and restart the application.

2. **Identify the root cause**
   Enable `echo_ffmpeg_logs = true` and capture the stderr output immediately preceding `ffmpeg exited with code 1`.

3. **Code fix**
   Add sequence reset detection to `m3u8_loader.py`.

4. **Improve robustness**
   Always log the stderr tail on abnormal `ffmpeg` exits and add timeout/partial-finalization handling when HLS stops during recording.

Based on the provided logs alone, the immediate failure is that **the RTSP-to-HLS `ffmpeg` process is repeatedly crashing**. The primary design flaw in the repository is that **the m3u8 loader cannot handle HLS sequence resets after an `ffmpeg` restart**, which directly leads to Slack uploads no longer occurring.

[1]: https://github.com/aont/event-recorder/blob/main/myrecorder/ffmpeg_utils.py "event-recorder/myrecorder/ffmpeg_utils.py at main · aont/event-recorder · GitHub"
[2]: https://github.com/aont/event-recorder/blob/main/myrecorder/recording.py "event-recorder/myrecorder/recording.py at main · aont/event-recorder · GitHub"
[3]: https://github.com/aont/event-recorder/blob/main/config.example.toml "event-recorder/config.example.toml at main · aont/event-recorder · GitHub"
[4]: https://ffmpeg.org/ffmpeg-formats.html "FFmpeg Formats Documentation"
[5]: https://github.com/aont/event-recorder/blob/main/myrecorder/m3u8_loader.py "event-recorder/myrecorder/m3u8_loader.py at main · aont/event-recorder · GitHub"
