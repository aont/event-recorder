from pathlib import Path

from myrecorder.ai_client import AiClient


def make_client(*, targets=None, unix_socket_path=None) -> AiClient:
    return AiClient(
        server_url="http://127.0.0.1:8080/",
        unix_socket_path=unix_socket_path,
        targets=targets or ["cat"],
        score_threshold=0.4,
        timeout_seconds=20.0,
        max_results=5,
    )


def test_ai_client_builds_detection_server_config() -> None:
    client = make_client(targets=["cat", "dog"])

    assert client._request_config() == {
        "object_detector_options": {
            "score_threshold": 0.4,
            "max_results": 5,
            "category_allowlist": ["cat", "dog"],
        }
    }
    assert client._detect_url() == "http://127.0.0.1:8080/v1/detect"


def test_ai_client_uses_localhost_url_for_unix_socket() -> None:
    client = make_client(unix_socket_path=Path("/tmp/detection-server.sock"))

    assert client._detect_url() == "http://localhost/v1/detect"


def test_ai_client_normalizes_detection_server_response() -> None:
    client = make_client(targets=["cat"])

    result = client._normalize_response(
        {
            "ok": True,
            "result": {
                "detections": [
                    {
                        "bounding_box": {"origin_x": 1, "origin_y": 2, "width": 3, "height": 4},
                        "categories": [
                            {"category_name": "cat", "score": 0.8},
                            {"category_name": "dog", "score": 0.9},
                            {"category_name": "cat", "score": 0.1},
                        ],
                    }
                ]
            },
        }
    )

    assert result == {
        "has_target": True,
        "num_targets": 1,
        "detections": [{"label": "cat", "score": 0.8, "bbox": {"x": 1, "y": 2, "width": 3, "height": 4}}],
    }
