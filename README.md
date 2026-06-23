# Recording System r4

Two-process reference implementation:

- `proc1`: RTSP → HLS with `asyncio.subprocess`, m3u8 loading, frame extraction, AI trigger handling, HLS clip capture, MP4 conversion, optional Slack upload.
- `proc2`: MediaPipe EfficientDet object detector served over an AF_UNIX socket.

The code targets Python 3.11+ on Linux/macOS. AF_UNIX sockets and hard links are POSIX features.

## Layout

```text
recording_system_r4/
  config.py              TOML config loader
  ffmpeg_utils.py        ffmpeg process helpers, HLS task, frame extraction
  hls.py                 lightweight HLS m3u8 parser/writer helpers
  ipc.py                 length-prefixed JSON AF_UNIX protocol
  ai_client.py           proc1 client for proc2
  m3u8_loader.py         source playlist loader and frame analyzer
  recording.py           triggered HLS capture, MP4 conversion, Slack upload
  proc1.py               proc1 entrypoint
  proc2.py               proc2 entrypoint
config.example.toml
requirements.txt
pyproject.toml
```

## Install

```bash
cd recording_system_r4
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e .
```

You also need an `ffmpeg` binary in `PATH`, or set `[hls].ffmpeg_bin`.

## Model file

Place `efficientdet_lite0.tflite` where `[proc2].model_path` points. The model is not bundled in this repository.

Example:

```bash
mkdir -p models
# put efficientdet_lite0.tflite at models/efficientdet_lite0.tflite
```

## Configure

Copy the example and edit at least these values:

```bash
cp config.example.toml config.toml
```

Minimum required changes:

```toml
[rtsp]
url = "rtsp://camera-or-nvr/stream"

[ai]
target_objects = ["cat"]
score_threshold = 0.4

[proc2]
model_path = "./models/efficientdet_lite0.tflite"
```

For Slack upload, configure explicit values in the TOML file:

```toml
[slack]
enabled = true
bot_token = "xoxb-..."
channel_id = "C0123456789"
```

The deprecated `bot_token_env` and `channel_id_env` settings are not read.

## Run

Start proc2 first:

```bash
recording-r4-proc2 --config config.toml
```

Start proc1 in another terminal:

```bash
recording-r4-proc1 --config config.toml
```

## Behavior

### proc1 ffmpeg HLS generation

`proc1` runs ffmpeg from the source HLS directory so segment URIs in the m3u8 are relative. The default HLS command uses:

```text
-f hls
-hls_time <segment_seconds>
-hls_list_size <retain_segments>
-hls_delete_threshold <delete_threshold>
-hls_start_number_source <start_number_source>
-hls_flags delete_segments+program_date_time+temp_file
```

The playlist is treated as the source of truth. FFmpeg stdout/stderr are still drained, and segment-like `Opening ...` log lines wake the m3u8 loader. By default, these ffmpeg logs are not echoed; set `[hls].echo_ffmpeg_logs = true` for debugging.


### m3u8-loader progress logging

By default, routine `m3u8-loader` progress logs are suppressed. This includes segment log wakeups, frame extraction start/completion messages, skipped-existing-segment messages, and frame-cleanup messages. Warnings, errors, and target-detection messages remain visible. To debug loader progress, set:

```toml
[frames]
echo_m3u8_progress_logs = true
```

### m3u8 loading and frame extraction

For each unprocessed segment, `proc1`:

1. Parses `#EXT-X-MEDIA-SEQUENCE` and segment `#EXTINF` durations.
2. Warns if two or more segments are unprocessed.
3. Extracts JPEG frames every `[frames].tc_seconds` using ffmpeg image2pipe.
4. Stores frame bytes in memory.
5. Optionally writes frame JPEGs to `[paths].frame_storage_dir`.
6. Sends each frame to proc2 over AF_UNIX.
7. Deletes derived frame images for segments no longer present in the source playlist when enabled.

Frame timestamps are derived from `#EXT-X-PROGRAM-DATE-TIME` plus the frame offset in the segment. If a segment lacks program date-time, the loader falls back to current UTC time for that segment.

### AI trigger and recording

If proc2 returns `has_target = true`, proc1 extends the active recording end time to:

```text
frame_timestamp + [frames].tb_seconds
```

If there is no active capture loop, proc1 starts one.

The recording task:

1. Creates a new directory under `[paths].recordings_dir`.
2. Hard-links current source segments into it. If hard-linking fails across filesystems, it falls back to `shutil.copy2` and logs a warning.
3. Copies the current m3u8 snapshot without `#EXT-X-ENDLIST`.
4. Polls the source m3u8 and appends new segment entries to the destination m3u8.
5. Ends the destination playlist with `#EXT-X-ENDLIST` after the copied segment timeline reaches the current end time.
6. Converts the captured HLS into MP4 with ffmpeg when `[recording].convert_to_mp4 = true`.
7. Uploads the MP4 to Slack when `[slack].enabled = true` and `bot_token` and `channel_id` are configured.
8. Deletes the per-recording directory, including copied segments and m3u8, after a successful Slack upload when `[recording].delete_dir_after_slack_upload = true` (default).


### Uploaded recording directory cleanup

By default, after a successful Slack upload, proc1 deletes the per-recording directory under `[paths].recordings_dir`. This removes the copied HLS playlist, hard-linked/copied segments, and generated MP4. Cleanup is not performed when Slack is disabled, `bot_token` or `channel_id` is missing, upload fails, or MP4 conversion fails.

To keep uploaded recording directories on disk:

```toml
[recording]
delete_dir_after_slack_upload = false
```

## AF_UNIX protocol

Messages are length-prefixed JSON:

```text
4-byte unsigned big-endian JSON length
UTF-8 JSON payload
```

`proc1` request shape:

```json
{
  "type": "analyze_frame",
  "request_id": "uuid",
  "frame_id": "seq123-offset000001000",
  "timestamp": "2026-06-21T12:00:00+00:00",
  "segment_sequence": 123,
  "segment_uri": "segment_0000000123.ts",
  "offset_seconds": 1.0,
  "targets": ["cat"],
  "score_threshold": 0.4,
  "image_format": "jpeg",
  "image_base64": "..."
}
```

`proc2` response shape:

```json
{
  "ok": true,
  "request_id": "uuid",
  "frame_id": "seq123-offset000001000",
  "has_target": true,
  "num_targets": 1,
  "detections": [
    {"label": "cat", "score": 0.91, "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}}
  ],
  "processing_ms": 32.5
}
```

## Notes and limitations

- This is a reference implementation. In this environment it was syntax-checked and parser smoke-tested, but not exercised against a real RTSP camera, ffmpeg binary, MediaPipe model, or Slack workspace.
- The default ffmpeg stream args use codec copy. Some RTSP streams require transcoding or bitstream filters; override `[hls].stream_args` / `[hls].output_args` as needed.
- The lightweight m3u8 parser covers the tags this system emits and consumes. It is not a full RFC 8216 parser.
- Hard links require source and destination to be on the same filesystem. The fallback copy is included to avoid losing recordings when deployment paths cross filesystem boundaries.
- `proc1` cleans `[paths].source_hls_dir` on startup by default. Disable with `[hls].clean_source_on_start = false` if you need to preserve that directory.


## FFmpeg log output

By default, `proc1` consumes ffmpeg stdout/stderr internally but does not echo ffmpeg log lines to the tool process stdout/stderr. This keeps runtime output quiet while still allowing segment-addition detection from ffmpeg logs. To debug ffmpeg output, set:

```toml
[hls]
echo_ffmpeg_logs = true
```
