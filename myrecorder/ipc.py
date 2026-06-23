from __future__ import annotations

import asyncio
import json
import struct
from typing import Any

_LENGTH_STRUCT = struct.Struct("!I")


class IpcProtocolError(RuntimeError):
    pass


async def read_json_message(reader: asyncio.StreamReader, *, max_bytes: int) -> dict[str, Any] | None:
    try:
        header = await reader.readexactly(_LENGTH_STRUCT.size)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise IpcProtocolError("truncated message length") from exc

    (length,) = _LENGTH_STRUCT.unpack(header)
    if length <= 0:
        raise IpcProtocolError("empty message")
    if length > max_bytes:
        raise IpcProtocolError(f"message too large: {length} > {max_bytes}")

    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError as exc:
        raise IpcProtocolError("truncated message payload") from exc

    try:
        message = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IpcProtocolError("invalid JSON payload") from exc

    if not isinstance(message, dict):
        raise IpcProtocolError("JSON payload must be an object")
    return message


async def write_json_message(writer: asyncio.StreamWriter, message: dict[str, Any]) -> None:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    writer.write(_LENGTH_STRUCT.pack(len(payload)))
    writer.write(payload)
    await writer.drain()
