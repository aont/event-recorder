# MediaPipe Unix Socket Server

This repository contains only the MediaPipe object-detection server from the previous recording system. It listens on an AF_UNIX socket, accepts length-prefixed JSON messages containing base64-encoded image bytes, and returns object detections.

The code targets Python 3.11+ on Linux/macOS. AF_UNIX sockets are POSIX features.

## Layout

```text
mediapipe_apiserver/
  ipc.py       length-prefixed JSON AF_UNIX protocol
  proc2.py     MediaPipe object detector socket server
requirements.txt
```

## Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Model file

Place a MediaPipe object detection `.tflite` model on disk. For example, use `efficientdet_lite0.tflite`.

## Run

All configuration is passed with command-line arguments:

```bash
python -m mediapipe_apiserver.proc2 \
  --socket-path /tmp/mediapipe-detector.sock \
  --model-path ./models/efficientdet_lite0.tflite \
  --target-object cat \
  --score-threshold 0.4
```

Useful options:

- `--socket-path`: Unix socket path to create.
- `--model-path`: MediaPipe `.tflite` model path.
- `--target-object`: allowed object label; repeat it to allow multiple labels. Defaults to `cat`.
- `--target-objects`: comma-separated allowed labels. This overrides repeated `--target-object` values.
- `--score-threshold`: minimum detection score. Defaults to `0.4`.
- `--max-results`: maximum MediaPipe results. Defaults to `-1`.
- `--max-message-bytes`: maximum IPC message size. Defaults to `20971520`.

## Protocol

Each message is a four-byte unsigned big-endian payload length followed by a UTF-8 JSON object.

Request:

```json
{
  "type": "analyze_frame",
  "request_id": "optional-id",
  "frame_id": "optional-frame-id",
  "timestamp": "optional timestamp",
  "targets": ["cat"],
  "score_threshold": 0.4,
  "image_base64": "..."
}
```

The per-request `targets` and `score_threshold` values are optional and default to the command-line configuration.

Response:

```json
{
  "ok": true,
  "request_id": "optional-id",
  "frame_id": "optional-frame-id",
  "timestamp": "optional timestamp",
  "targets": ["cat"],
  "score_threshold": 0.4,
  "processing_ms": 12.3,
  "has_target": true,
  "num_targets": 1,
  "detections": [
    {"label": "cat", "score": 0.91, "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}}
  ]
}
```
