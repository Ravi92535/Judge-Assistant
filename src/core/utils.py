"""
utils.py — Shared retry and delay utilities for the Judge Assistant pipeline.
"""

import time
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# ─── Groq rate-limit error class (safe import) ───────────────────────────────
try:
    import groq
    GroqRateLimitError = groq.RateLimitError
except (ImportError, AttributeError):
    class GroqRateLimitError(Exception):  # type: ignore[misc]
        """Placeholder when the groq package is not installed."""


def retry_on_quota(func):
    """
    Decorator that automatically retries the wrapped function when a Groq
    RateLimitError is raised, using exponential back-off (10 s → 120 s).
    Gives up after 10 attempts.
    """
    return retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=10, max=120),
        retry=retry_if_exception_type(GroqRateLimitError),
        before_sleep=lambda retry_state: print(
            f"Rate limit hit — retrying in "
            f"{retry_state.next_action.sleep:.0f}s "
            f"(attempt {retry_state.attempt_number}/10)..."
        ),
    )(func)


def delay_step(seconds: float = 2.0) -> None:
    """Small delay between sequential pipeline steps to avoid rate limits."""
    time.sleep(seconds)
