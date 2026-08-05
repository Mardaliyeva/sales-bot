from __future__ import annotations

import logging

from app.logging_config import configure_logging


def test_provider_http_loggers_do_not_emit_endpoint_urls_at_info() -> None:
    configure_logging("INFO")

    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING
    assert logging.getLogger("httpcore").getEffectiveLevel() == logging.WARNING
