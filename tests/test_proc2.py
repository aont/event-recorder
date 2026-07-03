import importlib
import sys
import types
from types import SimpleNamespace


def load_proc2_with_fake_mediapipe(monkeypatch):
    fake_mp = types.ModuleType("mediapipe")
    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.web = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "mediapipe", fake_mp)
    monkeypatch.setitem(sys.modules, "aiohttp", fake_aiohttp)
    sys.modules.pop("mediapipe_apiserver.proc2", None)
    return importlib.import_module("mediapipe_apiserver.proc2")


def test_detection_result_serialization_preserves_mediapipe_field_names(monkeypatch):
    proc2 = load_proc2_with_fake_mediapipe(monkeypatch)
    result = SimpleNamespace(
        detections=[
            SimpleNamespace(
                bounding_box=SimpleNamespace(origin_x=1, origin_y=2, width=3, height=4),
                categories=[
                    SimpleNamespace(index=17, score=0.91, display_name=None, category_name="cat"),
                ],
            )
        ]
    )

    assert proc2.detection_result_to_dict(result) == {
        "detections": [
            {
                "bounding_box": {"origin_x": 1, "origin_y": 2, "width": 3, "height": 4},
                "categories": [
                    {"index": 17, "score": 0.91, "display_name": None, "category_name": "cat"},
                ],
            }
        ]
    }
