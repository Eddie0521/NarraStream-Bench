"""Retry helpers for API-backed metric evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import random
import time


RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    initial_delay: float
    max_delay: float
    backoff: float
    jitter: float


def resolve_retry_policy(
    *,
    config=None,
    service_name: str,
    default_max_attempts: int = 12,
    default_initial_delay: float = 2.0,
    default_max_delay: float = 60.0,
    default_backoff: float = 2.0,
    default_jitter: float = 0.2,
) -> RetryPolicy:
    service_cfg = ((config or {}).get("services") or {}).get(service_name, {})
    retry_cfg = service_cfg.get("retry", {})
    return RetryPolicy(
        max_attempts=max(1, int(retry_cfg.get("max_attempts", default_max_attempts))),
        initial_delay=max(
            0.0,
            float(retry_cfg.get("initial_delay", default_initial_delay)),
        ),
        max_delay=max(0.0, float(retry_cfg.get("max_delay", default_max_delay))),
        backoff=max(1.0, float(retry_cfg.get("backoff", default_backoff))),
        jitter=max(0.0, float(retry_cfg.get("jitter", default_jitter))),
    )


def _status_code(exc: Exception) -> int | None:
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code

    response = getattr(exc, "response", None)
    candidate = getattr(response, "status_code", None)
    if isinstance(candidate, int):
        return candidate

    return None


def _retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value is None:
            continue
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        return max(0.0, seconds)
    return None


def _short_error_text(exc: Exception) -> str:
    text = " ".join(str(exc).split())
    return text[:240] if text else exc.__class__.__name__


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True

    status_code = _status_code(exc)
    if status_code in RETRYABLE_STATUS_CODES:
        return True

    lowered = _short_error_text(exc).casefold()
    retry_markers = (
        "rate limit",
        "rate-limit",
        "tpm limit",
        "too many requests",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection error",
        "connection reset",
        "service unavailable",
        "server error",
        "try again",
    )
    return any(marker in lowered for marker in retry_markers)


def _compute_delay(exc: Exception, attempt: int, policy: RetryPolicy) -> float:
    delay = policy.initial_delay * (policy.backoff ** max(0, attempt - 1))
    delay = min(delay, policy.max_delay)

    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        delay = max(delay, retry_after)

    if policy.jitter > 0:
        jitter_ratio = random.uniform(1.0 - policy.jitter, 1.0 + policy.jitter)
        delay *= jitter_ratio

    return max(0.0, delay)


def run_with_retry(
    operation,
    *,
    label: str,
    policy: RetryPolicy,
    retryable=is_retryable_exception,
):
    attempt = 1
    while True:
        try:
            return operation()
        except Exception as exc:
            can_retry = attempt < policy.max_attempts and retryable(exc)
            short_error = _short_error_text(exc)
            if not can_retry:
                print(
                    f"[retry] {label}: giving up after {attempt}/{policy.max_attempts} "
                    f"attempts: {short_error}"
                )
                raise

            delay = _compute_delay(exc, attempt, policy)
            print(
                f"[retry] {label}: attempt {attempt}/{policy.max_attempts} failed: "
                f"{short_error} | sleeping {delay:.1f}s"
            )
            time.sleep(delay)
            attempt += 1
