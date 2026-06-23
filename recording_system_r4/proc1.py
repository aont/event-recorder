from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path

from .ai_client import AiClient
from .config import AppConfig
from .ffmpeg_utils import FfmpegHlsTask, SegmentLogEvent, utc_stamp
from .m3u8_loader import M3u8LoadingTask
from .recording import RecordingManager


def clean_source_hls_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    print(f"{utc_stamp()} proc1: cleaned source HLS dir {path}", file=sys.stderr, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recording System r4 proc1: RTSP→HLS, frame analysis, recording")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    return parser.parse_args(argv)


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AppConfig.from_file(args.config)
    config.validate_proc1()
    config.ensure_directories()

    if config.hls.clean_source_on_start:
        clean_source_hls_dir(config.paths.source_hls_dir)

    if not config.ai.socket_path.exists():
        print(
            f"{utc_stamp()} proc1: warning: AI socket does not exist yet: {config.ai.socket_path}. "
            "Start proc2 before proc1.",
            file=sys.stderr,
            flush=True,
        )

    segment_events: asyncio.Queue[SegmentLogEvent] = asyncio.Queue()
    ai_client = AiClient(
        socket_path=config.ai.socket_path,
        targets=config.ai.target_objects,
        score_threshold=config.ai.score_threshold,
        timeout_seconds=config.ai.timeout_seconds,
        max_message_bytes=config.ai.max_message_bytes,
    )
    recording_manager = RecordingManager(config)
    ffmpeg_task_runner = FfmpegHlsTask(config, segment_events)
    m3u8_loader = M3u8LoadingTask(
        config=config,
        segment_events=segment_events,
        ai_client=ai_client,
        recording_manager=recording_manager,
    )

    ffmpeg_task = asyncio.create_task(ffmpeg_task_runner.run(), name="ffmpeg-hls")
    loader_task = asyncio.create_task(m3u8_loader.run(), name="m3u8-loader")

    try:
        done, pending = await asyncio.wait({ffmpeg_task, loader_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
        if ffmpeg_task in done:
            rc = ffmpeg_task.result()
            if rc != 0:
                return rc
            return 0
        # loader returned normally, which should not happen.
        return 1
    finally:
        loader_task.cancel()
        await ffmpeg_task_runner.terminate()
        if not ffmpeg_task.done():
            ffmpeg_task.cancel()
        await asyncio.gather(ffmpeg_task, loader_task, return_exceptions=True)
        await recording_manager.close()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
