from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from .config import SlackConfig
from .ffmpeg_utils import utc_stamp


async def upload_file_to_slack(config: SlackConfig, file_path: Path, *, title: str) -> bool:
    if not config.enabled:
        print(f"{utc_stamp()} slack: disabled; not uploading {file_path}", file=sys.stderr, flush=True)
        return False

    token = config.resolved_token()
    channel_id = config.resolved_channel_id()
    if not token or not channel_id:
        print(
            f"{utc_stamp()} slack: enabled but token/channel_id is missing; not uploading {file_path}",
            file=sys.stderr,
            flush=True,
        )
        return False

    def _upload() -> None:
        try:
            from slack_sdk import WebClient
            from slack_sdk.errors import SlackApiError
        except ImportError as exc:
            raise RuntimeError("slack-sdk is not installed") from exc

        client = WebClient(token=token)
        try:
            client.files_upload_v2(
                channel=channel_id,
                file=str(file_path),
                title=title,
                initial_comment=config.initial_comment,
            )
        except SlackApiError as exc:
            response = getattr(exc, "response", None)
            raise RuntimeError(f"Slack upload failed: {response or exc}") from exc

    await asyncio.to_thread(_upload)
    print(f"{utc_stamp()} slack: uploaded {file_path}", file=sys.stderr, flush=True)
    return True
