"""Bounded, retrying HTTP transport shared by live provider adapters."""

from __future__ import annotations

import asyncio
import email.utils
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import httpx

from oasis.providers.models import CancellationSignal, ProviderError, ProviderErrorCode
from oasis.providers.redaction import ProviderLogger, RedactingLogger, redact_url


class NullProviderLogger:
    """No-op logger used when an embedding does not request provider logs."""

    def info(self, message: str, **context: object) -> None:
        del message, context

    def warning(self, message: str, **context: object) -> None:
        del message, context


class AuthenticationHook(Protocol):
    """Apply provider credentials without exposing them to request models or provenance."""

    def apply(self, headers: dict[str, str], parameters: dict[str, str]) -> None: ...


@dataclass(frozen=True, slots=True)
class NoAuthentication:
    """No-op authentication hook for public endpoints."""

    def apply(self, headers: dict[str, str], parameters: dict[str, str]) -> None:
        del headers, parameters


@dataclass(frozen=True, slots=True, repr=False)
class BearerTokenAuthentication:
    """Bearer authentication whose repr never contains the token."""

    token: str

    def __repr__(self) -> str:
        return "BearerTokenAuthentication(token='[REDACTED]')"

    def apply(self, headers: dict[str, str], parameters: dict[str, str]) -> None:
        del parameters
        headers["Authorization"] = f"Bearer {self.token}"


@dataclass(frozen=True, slots=True, repr=False)
class ApiKeyAuthentication:
    """Header or query API-key hook whose repr never contains the credential."""

    name: str
    value: str
    in_query: bool = False

    def __repr__(self) -> str:
        return (
            f"ApiKeyAuthentication(name={self.name!r}, value='[REDACTED]', "
            f"in_query={self.in_query!r})"
        )

    def apply(self, headers: dict[str, str], parameters: dict[str, str]) -> None:
        target = parameters if self.in_query else headers
        target[self.name] = self.value


@dataclass(frozen=True, slots=True)
class HttpPolicy:
    """Network safety and resilience limits for all requests made by one client."""

    user_agent: str = "oasis-anytime-agents/0.1 (configure OASIS_PROVIDER_USER_AGENT)"
    timeout_seconds: float = 10.0
    max_attempts: int = 3
    backoff_base_seconds: float = 0.25
    max_response_bytes: int = 10_000_000
    max_pages: int = 20

    def __post_init__(self) -> None:
        if not self.user_agent.strip():
            raise ValueError("provider user agent cannot be empty")
        if self.timeout_seconds <= 0:
            raise ValueError("provider timeout must be positive")
        if self.max_attempts < 1:
            raise ValueError("provider max attempts must be positive")
        if self.backoff_base_seconds < 0:
            raise ValueError("provider backoff cannot be negative")
        if self.max_response_bytes < 1:
            raise ValueError("provider response size limit must be positive")
        if self.max_pages < 1:
            raise ValueError("provider page limit must be positive")


@dataclass(frozen=True, slots=True)
class HttpPayload:
    """A fully read response bounded by the configured byte limit."""

    status_code: int
    headers: Mapping[str, str]
    content: bytes
    url: str

    def json(self) -> object:
        try:
            return httpx.Response(self.status_code, content=self.content).json()
        except ValueError as error:
            raise ProviderError(
                ProviderErrorCode.MALFORMED_RESPONSE,
                "provider returned malformed JSON",
            ) from error


Sleep = Callable[[float], Awaitable[None]]


class ResilientHttpClient:
    """HTTPX adapter enforcing deadlines, byte limits, retries, and safe diagnostics."""

    _RETRY_STATUSES = frozenset({408, 425, 500, 502, 503, 504})

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        policy: HttpPolicy | None = None,
        authentication: AuthenticationHook | None = None,
        sleep: Sleep = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        logger: ProviderLogger | None = None,
    ) -> None:
        self.client = client
        self.policy = policy or HttpPolicy()
        self.authentication = authentication or NoAuthentication()
        self._sleep = sleep
        self._monotonic = monotonic
        self._logger = RedactingLogger(logger or NullProviderLogger())

    @staticmethod
    def _retry_after(value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = email.utils.parsedate_to_datetime(value)
            except (TypeError, ValueError):
                return None
            return max(0.0, (parsed - datetime.now(UTC)).total_seconds())

    async def _backoff(
        self,
        attempt: int,
        *,
        deadline_monotonic: float,
        retry_after: float | None = None,
    ) -> None:
        delay = (
            retry_after
            if retry_after is not None
            else self.policy.backoff_base_seconds * (2 ** (attempt - 1))
        )
        if delay <= 0:
            return
        remaining = deadline_monotonic - self._monotonic()
        if remaining <= delay:
            raise ProviderError(ProviderErrorCode.TIMEOUT, "provider deadline expired")
        await self._sleep(delay)

    async def request(
        self,
        method: str,
        url: str,
        *,
        deadline_monotonic: float,
        cancellation: CancellationSignal,
        parameters: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        json_body: object | None = None,
        redact_parameter_names: frozenset[str] = frozenset(),
        safe_url_override: str | None = None,
    ) -> HttpPayload:
        """Return bounded bytes or raise a typed provider failure after bounded retries."""

        request_headers = {"User-Agent": self.policy.user_agent, "Accept": "application/json"}
        request_headers.update(headers or {})
        request_parameters = dict(parameters or {})
        self.authentication.apply(request_headers, request_parameters)
        safe_url = safe_url_override or redact_url(
            str(httpx.URL(url, params=request_parameters)), redact_parameter_names
        )

        for attempt in range(1, self.policy.max_attempts + 1):
            cancellation.raise_if_cancelled()
            remaining = deadline_monotonic - self._monotonic()
            if remaining <= 0:
                raise ProviderError(ProviderErrorCode.TIMEOUT, "provider deadline expired")
            timeout = min(self.policy.timeout_seconds, remaining)
            self._logger.info(
                "provider request",
                method=method,
                url=safe_url,
                attempt=attempt,
            )
            try:
                async with self.client.stream(
                    method,
                    url,
                    params=request_parameters,
                    headers=request_headers,
                    json=json_body,
                    timeout=timeout,
                ) as response:
                    declared_length = response.headers.get("Content-Length")
                    if (
                        declared_length is not None
                        and declared_length.isdigit()
                        and int(declared_length) > self.policy.max_response_bytes
                    ):
                        raise ProviderError(
                            ProviderErrorCode.RESPONSE_TOO_LARGE,
                            "provider response exceeded the configured size limit",
                        )
                    chunks: list[bytes] = []
                    size = 0
                    async for chunk in response.aiter_bytes():
                        cancellation.raise_if_cancelled()
                        size += len(chunk)
                        if size > self.policy.max_response_bytes:
                            raise ProviderError(
                                ProviderErrorCode.RESPONSE_TOO_LARGE,
                                "provider response exceeded the configured size limit",
                            )
                        chunks.append(chunk)
                    status = response.status_code
                    payload = b"".join(chunks)
                    response_headers = dict(response.headers)
                    response_url = safe_url_override or redact_url(
                        str(response.url), redact_parameter_names
                    )
            except ProviderError:
                raise
            except httpx.TimeoutException as error:
                if attempt == self.policy.max_attempts:
                    raise ProviderError(
                        ProviderErrorCode.TIMEOUT,
                        "provider request timed out",
                        retryable=True,
                    ) from error
                await self._backoff(attempt, deadline_monotonic=deadline_monotonic)
                continue
            except httpx.RequestError as error:
                if attempt == self.policy.max_attempts:
                    raise ProviderError(
                        ProviderErrorCode.UNAVAILABLE,
                        "provider request failed",
                        retryable=True,
                    ) from error
                await self._backoff(attempt, deadline_monotonic=deadline_monotonic)
                continue

            if status == 429:
                retry_after = self._retry_after(response_headers.get("retry-after"))
                if attempt < self.policy.max_attempts:
                    await self._backoff(
                        attempt,
                        deadline_monotonic=deadline_monotonic,
                        retry_after=retry_after,
                    )
                    continue
                raise ProviderError(
                    ProviderErrorCode.RATE_LIMITED,
                    "provider rate limit was exhausted",
                    retryable=True,
                    retry_after_seconds=retry_after,
                )
            if status in self._RETRY_STATUSES and attempt < self.policy.max_attempts:
                await self._backoff(attempt, deadline_monotonic=deadline_monotonic)
                continue
            if status >= 400:
                raise ProviderError(
                    ProviderErrorCode.UNAVAILABLE,
                    f"provider returned HTTP {status}",
                    retryable=status in self._RETRY_STATUSES,
                )
            return HttpPayload(
                status_code=status,
                headers=response_headers,
                content=payload,
                url=response_url,
            )
        raise AssertionError("HTTP retry loop must return or raise")
