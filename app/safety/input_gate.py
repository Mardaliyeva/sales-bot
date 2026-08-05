from __future__ import annotations

import unicodedata


class InputRejected(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def validate_and_clean_message(message: str, *, max_length: int = 4000) -> str:
    normalized = unicodedata.normalize("NFC", message)
    cleaned = "".join(
        char for char in normalized if char in {"\n", "\r", "\t"} or unicodedata.category(char) != "Cc"
    ).strip()
    if not cleaned:
        raise InputRejected("empty_message", "Mesaj boş ola bilməz.")
    if len(cleaned) > max_length:
        raise InputRejected(
            "message_too_long",
            f"Mesaj maksimum {max_length} simvol ola bilər.",
        )
    return cleaned
