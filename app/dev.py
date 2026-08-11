from __future__ import annotations

import argparse
import asyncio
import selectors
import socket
import sys
from collections.abc import Sequence

import uvicorn

from app.config import get_settings


def create_windows_selector_event_loop() -> asyncio.AbstractEventLoop:
    return asyncio.SelectorEventLoop(selectors.SelectSelector())


def listener_exists(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.25)
        return probe.connect_ex((host, port)) == 0


def main(argv: Sequence[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Windows-safe Sales Bot API dev launcher")
    parser.add_argument("--host", default=settings.app_host)
    parser.add_argument("--port", type=int, default=settings.app_port)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)
    if listener_exists(args.host, args.port):
        print(f"API artıq {args.host}:{args.port} ünvanında işləyir; ikinci instansiya açılmadı.")
        return 0

    loop: str | object = (
        create_windows_selector_event_loop if sys.platform == "win32" else "auto"
    )
    config = uvicorn.Config(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop=loop,  # type: ignore[arg-type]
        log_level=settings.log_level.casefold(),
    )
    uvicorn.Server(config).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
