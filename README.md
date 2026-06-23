# Event Recorder

A Python RTSP camera recorder that converts streams to HLS, detects target objects with MediaPipe EfficientDet, and records event clips. When an event is detected, it can capture video retroactively from before the detection time, convert clips to MP4, and optionally upload them to Slack.

Single-command implementation:

- `myrecorder`: RTSP → HLS with `asyncio.subprocess`, m3u8 loading, frame extraction, AI trigger handling, HLS clip capture, MP4 conversion, optional Slack upload.
- MediaPipe EfficientDet inference runs from `myrecorder` through `concurrent.futures.ProcessPoolExecutor`.

The code targets Python 3.11+ on Linux/macOS. Hard links are POSIX features.

## Layout

```text
myrecorder/
  config.py              TOML config loader
  ffmpeg_utils.py        ffmpeg process helpers, HLS task, frame extraction
  hls.py                 lightweight HLS m3u8 parser/writer helpers
  ai_worker.py           MediaPipe detector code executed in worker processes
  ai_client.py           asyncio facade over ProcessPoolExecutor
  m3u8_loader.py         source playlist loader and frame analyzer
  recording.py           triggered HLS capture, MP4 conversion, Slack upload
  app.py                 myrecorder entrypoint
config.example.toml
requirements.txt
pyproject.toml
```

`ipc.py` may remain in older source trees for compatibility tests, but runtime no longer uses AF_UNIX.

## Setup

From the repository root, create a virtual environment and install the project package with its runtime dependencies:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install .
```

Installing the package creates the `myrecorder` console script inside the active virtualenv. Run it with:

```bash
myrecorder --config config.toml
```

You also need an `ffmpeg` binary in `PATH`, or set `[hls].ffmpeg_bin`.

## Setup without installing as a package

From the repository root, create a virtual environment and install only runtime dependencies:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Then run either of these from the repository root:

```bash
python3 -m myrecorder --config config.toml
```

or:

```bash
./bin/myrecorder --config config.toml
```

You also need an `ffmpeg` binary in `PATH`, or set `[hls].ffmpeg_bin`.

## Model file

Place `efficientdet_lite0.tflite` where `[ai].model_path` points. The model is not bundled in this repository.

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
model_path = "./models/efficientdet_lite0.tflite"
target_objects = ["cat"]
score_threshold = 0.4
workers = 1
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

Without installing the package:

```bash
python3 -m myrecorder --config config.toml
```

or:

```bash
./bin/myrecorder --config config.toml
```

After optional `pip install .`, the console script is also available:

```bash
myrecorder --config config.toml
```

## Behavior

### ffmpeg HLS generation

`myrecorder` runs ffmpeg from the source HLS directory so segment URIs in the m3u8 are relative. The default HLS command uses:

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

For each unprocessed segment, `myrecorder`:

1. Parses `#EXT-X-MEDIA-SEQUENCE` and segment `#EXTINF` durations.
2. Warns if two or more segments are unprocessed.
3. Extracts JPEG frames every `[frames].tc_seconds` using ffmpeg image2pipe.
4. Stores frame bytes in memory.
5. Optionally writes frame JPEGs to `[paths].frame_storage_dir`.
6. Submits each frame to a `ProcessPoolExecutor` MediaPipe worker through an asyncio-compatible facade.
7. Deletes derived frame images for segments no longer present in the source playlist when enabled.

Frame timestamps are derived from `#EXT-X-PROGRAM-DATE-TIME` plus the frame offset in the segment. If a segment lacks program date-time, the loader falls back to current UTC time for that segment.

### AI execution model

`myrecorder` creates a `concurrent.futures.ProcessPoolExecutor` with `[ai].workers` worker processes. Each worker initializes its own MediaPipe `ObjectDetector` from `[ai].model_path` and caches detectors by target/threshold.

The main process remains asyncio-driven. Frame analysis is awaited with:

```text
asyncio event loop -> run_in_executor(ProcessPoolExecutor, MediaPipe detection)
```

If detection exceeds `[ai].timeout_seconds`, the frame request is treated as failed and the recorder continues.

### AI trigger and recording

If analysis returns `has_target = true`, `myrecorder` extends the active recording end time to:

```text
frame_timestamp + [frames].tb_seconds
```

If there is no active capture loop, `myrecorder` starts one.

The recording task:

1. Creates a new timestamped directory under `[paths].recordings_dir`. By default, the timestamp uses the host process' local timezone and includes its numeric UTC offset.
2. Hard-links current source segments into it. If hard-linking fails across filesystems, it falls back to `shutil.copy2` and logs a warning.
3. Copies the current m3u8 snapshot without `#EXT-X-ENDLIST`.
4. Polls the source m3u8 and appends new segment entries to the destination m3u8.
5. Ends the destination playlist with `#EXT-X-ENDLIST` after the copied segment timeline reaches the current end time.
6. Converts the captured HLS into MP4 with ffmpeg when `[recording].convert_to_mp4 = true`.
7. Uploads the MP4 to Slack when `[slack].enabled = true` and `bot_token` and `channel_id` are configured.
8. Deletes the per-recording directory, including copied segments and m3u8, after a successful Slack upload when `[recording].delete_dir_after_slack_upload = true` (default).

### Uploaded recording directory cleanup

By default, after a successful Slack upload, `myrecorder` deletes the per-recording directory under `[paths].recordings_dir`. This removes the copied HLS playlist, hard-linked/copied segments, and generated MP4. Cleanup is not performed when Slack is disabled, `bot_token` or `channel_id` is missing, upload fails, or MP4 conversion fails.

To keep uploaded recording directories on disk:

```toml
[recording]
delete_dir_after_slack_upload = false
```

### Timestamped recording names

Recording directory names are generated as:

```text
rec_YYYYMMDDTHHMMSS.ffffff+ZZZZ_NNNN
```

By default, `YYYYMMDDTHHMMSS` is based on the host process' local timezone, for example `+0900` on a machine configured for Japan Standard Time. To use the previous UTC filename form ending in `Z`:

```toml
[recording]
use_local_time_for_filenames = false
```

## FFmpeg log output

By default, `myrecorder` consumes ffmpeg stdout/stderr internally but does not echo ffmpeg log lines to the tool process stdout/stderr. This keeps runtime output quiet while still allowing segment-addition detection from ffmpeg logs. To debug ffmpeg output, set:

```toml
[hls]
echo_ffmpeg_logs = true
```

## Notes and limitations

- This is a reference implementation. In this environment it was syntax-checked and parser smoke-tested, but not exercised against a real RTSP camera, ffmpeg binary, MediaPipe model, or Slack workspace.
- The default ffmpeg stream args use codec copy. Some RTSP streams require transcoding or bitstream filters; override `[hls].stream_args` / `[hls].output_args` as needed.
- The lightweight m3u8 parser covers the tags this system emits and consumes. It is not a full RFC 8216 parser.
- Hard links require source and destination to be on the same filesystem. The fallback copy is included to avoid losing recordings when deployment paths cross filesystem boundaries.
- `myrecorder` cleans `[paths].source_hls_dir` on startup by default. Disable with `[hls].clean_source_on_start = false` if you need to preserve that directory.
