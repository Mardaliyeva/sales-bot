from __future__ import annotations

import pytest

from app.safety.input_gate import InputRejected, validate_and_clean_message


def test_empty_message_is_rejected() -> None:
    with pytest.raises(InputRejected, match="boş"):
        validate_and_clean_message(" \t\n ")


def test_message_longer_than_limit_is_rejected() -> None:
    with pytest.raises(InputRejected) as captured:
        validate_and_clean_message("a" * 4001)
    assert captured.value.code == "message_too_long"


def test_control_characters_are_removed() -> None:
    assert validate_and_clean_message(" Salam\x00\x01\n dünya ") == "Salam\n dünya"
