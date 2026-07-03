# MediaPipe HTTP API Server

This repository contains the MediaPipe object-detection server from the previous recording system. It exposes an HTTP JSON API, accepts base64-encoded image bytes, and returns object detections.

The code targets Python 3.11+ and uses `aiohttp` for the HTTP server. MediaPipe work is dispatched to a dedicated single-worker thread so the asyncio event loop can continue accepting HTTP requests while detection is running.

## Layout

```text
mediapipe_apiserver/
  ipc.py       legacy length-prefixed JSON helpers
  proc2.py     MediaPipe object detector HTTP API server
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
  --host 127.0.0.1 \
  --port 8080 \
  --model-path ./models/efficientdet_lite0.tflite \
  --target-object cat \
  --score-threshold 0.4
```

Useful options:

- `--host`: HTTP bind host/IP address. Defaults to `127.0.0.1`.
- `--port`: HTTP bind port. Defaults to `8080`.
- `--model-path`: MediaPipe `.tflite` model path.
- `--target-object`: allowed object label; repeat it to allow multiple labels. Defaults to `cat`.
- `--target-objects`: comma-separated allowed labels. This overrides repeated `--target-object` values.
- `--score-threshold`: minimum detection score. Defaults to `0.4`.
- `--max-results`: maximum MediaPipe results. Defaults to `-1`.
- `--max-request-bytes`: maximum HTTP request body size. Defaults to `20971520`.

## API

### `GET /health`

Health check endpoint.

Response:

```json
{"ok": true}
```

### `POST /v1/analyze-frame`

Analyze one image. Send `Content-Type: application/json` with a JSON object body.

Request:

```json
{
  "request_id": "optional-id",
  "frame_id": "optional-frame-id",
  "timestamp": "optional timestamp",
  "targets": ["cat"],
  "score_threshold": 0.4,
  "image_base64": "..."
}
```

The per-request `targets` and `score_threshold` values are optional and default to the command-line configuration.

Successful response (`200`):

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

Error responses use `4xx` for bad requests and `5xx` for server-side detection failures:

```json
{"ok": false, "request_id": "optional-id", "error": "image_base64 is required"}
```

## Client examples

### curl

```bash
IMAGE_BASE64=$(python - <<'PY'
import base64
from pathlib import Path
print(base64.b64encode(Path('frame.jpg').read_bytes()).decode())
PY
)

curl -sS http://127.0.0.1:8080/v1/analyze-frame \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg image "$IMAGE_BASE64" '{request_id:"curl-example", targets:["cat"], image_base64:$image}')"
```

If `jq` is not available, generate the JSON with Python:

```bash
python - <<'PY' | curl -sS http://127.0.0.1:8080/v1/analyze-frame \
  -H 'Content-Type: application/json' \
  -d @-
import base64
import json
from pathlib import Path

print(json.dumps({
    "request_id": "curl-python-example",
    "targets": ["cat"],
    "image_base64": base64.b64encode(Path("frame.jpg").read_bytes()).decode(),
}))
PY
```

### JavaScript fetch

```js
const file = document.querySelector('input[type="file"]').files[0];
const dataUrl = await new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = reject;
  reader.readAsDataURL(file);
});

const response = await fetch('http://127.0.0.1:8080/v1/analyze-frame', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    request_id: crypto.randomUUID(),
    targets: ['cat'],
    image_base64: dataUrl.split(',', 2)[1],
  }),
});

const result = await response.json();
console.log(result);
```
