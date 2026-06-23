from datetime import datetime, timezone

from recording_system_r4.recording import _safe_timestamp


def test_safe_timestamp_utc_legacy_form():
    dt = datetime(2026, 6, 21, 1, 2, 3, 456789, tzinfo=timezone.utc)
    assert _safe_timestamp(dt, use_local_time=False) == "20260621T010203.456789Z"


def test_safe_timestamp_local_form_contains_numeric_offset():
    dt = datetime(2026, 6, 21, 1, 2, 3, 456789, tzinfo=timezone.utc)
    value = _safe_timestamp(dt, use_local_time=True)
    assert value.startswith("20260621T") or value.startswith("20260620T") or value.startswith("20260622T")
    assert value[-5] in "+-"
    assert value[-4:].isdigit()
