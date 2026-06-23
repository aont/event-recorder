import asyncio

from recording_system_r4.ipc import read_json_message, write_json_message


class MemoryWriter:
    def __init__(self):
        self.data = bytearray()

    def write(self, data):
        self.data.extend(data)

    async def drain(self):
        return None


def test_length_prefixed_json_roundtrip():
    async def run():
        writer = MemoryWriter()
        await write_json_message(writer, {"ok": True, "n": 1})
        reader = asyncio.StreamReader()
        reader.feed_data(bytes(writer.data))
        reader.feed_eof()
        return await read_json_message(reader, max_bytes=1024)

    assert asyncio.run(run()) == {"ok": True, "n": 1}
