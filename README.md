# MediaPipe HTTP API Server

This repository contains the MediaPipe object-detection server from the previous recording system. It exposes an HTTP API, accepts multipart image uploads with configuration data, and returns object detections.

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

Analyze one image. Send `Content-Type: multipart/form-data` with the raw image bytes in an `image` part and configuration in either a JSON `config` part or individual form fields.

Multipart parts:

- `image` (required): image file bytes.
- `config` (optional): JSON object with request metadata and detection settings.
- Individual fields (optional): `request_id`, `frame_id`, `timestamp`, `targets`, and `score_threshold`. Individual fields are merged with the `config` object as they are read.

Example `config` object:

```json
{
  "request_id": "optional-id",
  "frame_id": "optional-frame-id",
  "timestamp": "optional timestamp",
  "targets": ["cat"],
  "score_threshold": 0.4
}
```

The per-request `targets` and `score_threshold` values are optional and default to the command-line configuration. `targets` may be sent as a JSON string array when using an individual form field. These values are echoed for request compatibility only; detections are returned from MediaPipe without server-side filtering, relabeling, score-threshold processing, or bounding-box renaming.

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
  "detections": [
    {
      "bounding_box": {"origin_x": 1, "origin_y": 2, "width": 3, "height": 4},
      "categories": [
        {"index": 17, "score": 0.91, "display_name": null, "category_name": "cat"}
      ]
    }
  ]
}
```

Error responses use `4xx` for bad requests and `5xx` for server-side detection failures:

```json
{"ok": false, "request_id": "optional-id", "error": "image multipart field is required"}
```

## Client examples

### curl

```bash
curl -sS http://127.0.0.1:8080/v1/analyze-frame \
  -F 'image=@frame.jpg' \
  -F 'config={"request_id":"curl-example","targets":["cat"],"score_threshold":0.4};type=application/json'
```

You can also send configuration as individual multipart fields:

```bash
curl -sS http://127.0.0.1:8080/v1/analyze-frame \
  -F 'image=@frame.jpg' \
  -F 'request_id=curl-fields-example' \
  -F 'targets=["cat"]' \
  -F 'score_threshold=0.4'
```

### JavaScript fetch

```js
const file = document.querySelector('input[type="file"]').files[0];
const formData = new FormData();
formData.append('image', file, file.name);
formData.append('config', new Blob([JSON.stringify({
  request_id: crypto.randomUUID(),
  targets: ['cat'],
  score_threshold: 0.4,
})], { type: 'application/json' }));

const response = await fetch('http://127.0.0.1:8080/v1/analyze-frame', {
  method: 'POST',
  body: formData,
});

const result = await response.json();
console.log(result);
```
