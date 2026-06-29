from __future__ import annotations

import argparse
import asyncio
import sys
from .ai_client import AiClient
from .config import AppConfig
from .ffmpeg_utils import FfmpegHlsTask, SegmentLogEvent, clean_source_hls_dir
from .m3u8_loader import M3u8LoadingTask
from .recording import RecordingManager


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="myrecorder: RTSP→HLS, frame analysis, recording")
    parser.add_argument("--config", required=True, help="Path to TOML config")
    return parser.parse_args(argv)


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = AppConfig.from_file(args.config)
    config.validate_app()
    config.ensure_directories()

    if config.hls.clean_source_on_start:
        clean_source_hls_dir(config.paths.source_hls_dir)

    segment_events: asyncio.Queue[SegmentLogEvent] = asyncio.Queue()
    ai_client = AiClient(
        model_path=config.ai.model_path,
        targets=config.ai.target_objects,
        score_threshold=config.ai.score_threshold,
        timeout_seconds=config.ai.timeout_seconds,
        max_results=config.ai.max_results,
        workers=config.ai.workers,
    )
    ai_client.start()
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
        await ai_client.close()
        await recording_manager.close()


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    main()
