import time
from datetime import datetime
from zoneinfo import ZoneInfo

from myrecorder.ffmpeg_utils import log_stamp


def test_log_stamp_uses_host_local_timezone(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()

    value = log_stamp()

    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == datetime.now(ZoneInfo("Asia/Tokyo")).utcoffset()
    assert value.endswith("+09:00")
