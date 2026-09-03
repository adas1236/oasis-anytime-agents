"""Secret-safe URL, mapping, and logging helpers for provider integrations."""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from typing import Final, Protocol
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class ProviderLogger(Protocol):
    """Minimal structured logger accepted without depending on the tool layer."""

    def info(self, message: str, **context: object) -> None: ...

    def warning(self, message: str, **context: object) -> None: ...


REDACTED: Final = "[REDACTED]"
_SENSITIVE_KEY: Final = re.compile(
    r"(?:authorization|api[-_]?key|access[-_]?token|token|secret|password|signature|sig)",
    re.IGNORECASE,
)


def is_sensitive_key(key: str) -> bool:
    """Return whether a header/query key conventionally carries a credential."""

    return _SENSITIVE_KEY.search(key) is not None


def redact_mapping(values: Mapping[str, object]) -> dict[str, object]:
    """Copy a mapping while removing values associated with credential-like keys."""

    return {key: REDACTED if is_sensitive_key(key) else value for key, value in values.items()}


def redact_url(url: str, sensitive_keys: Collection[str] = ()) -> str:
    """Remove credentials and sensitive query values from a URL used in provenance or logs."""

    split = urlsplit(url)
    hostname = split.hostname or ""
    if split.port is not None:
        hostname = f"{hostname}:{split.port}"
    explicit = {key.casefold() for key in sensitive_keys}
    query = urlencode(
        [
            (key, REDACTED if is_sensitive_key(key) or key.casefold() in explicit else value)
            for key, value in parse_qsl(split.query)
        ]
    )
    return urlunsplit((split.scheme, hostname, split.path, query, ""))


class RedactingLogger:
    """Tool logger wrapper that redacts secret-like structured context recursively."""

    def __init__(self, logger: ProviderLogger) -> None:
        self._logger = logger

    @staticmethod
    def _clean(value: object) -> object:
        if isinstance(value, str) and "://" in value:
            return redact_url(value)
        if isinstance(value, Mapping):
            return {
                str(key): REDACTED if is_sensitive_key(str(key)) else RedactingLogger._clean(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [RedactingLogger._clean(item) for item in value]
        return value

    def info(self, message: str, **context: object) -> None:
        self._logger.info(message, **{key: self._clean(value) for key, value in context.items()})

    def warning(self, message: str, **context: object) -> None:
        self._logger.warning(message, **{key: self._clean(value) for key, value in context.items()})
