"""
utils/llm_client.py
Groq LLM client with retry logic, rate limit handling, and error reporting.
"""

import os
import re
import time
import logging
from typing import Optional, List, Dict
from groq import Groq, APIError, RateLimitError, APIConnectionError, AuthenticationError

from dotenv import load_dotenv

load_dotenv(override=True)


logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama-3.1-8b-instant"
MAX_RETRIES = 6
BASE_BACKOFF = 8.0
MIN_REQUEST_INTERVAL = 4.0
MAX_RETRY_WAIT = 90.0


class LLMClient:
    """Groq LLM client with production-grade error handling."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it in your .env file or environment."
            )
        self.client = Groq(api_key=key)
        self.model = model or DEFAULT_MODEL

    def chat(
        self,
        messages: List[Dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Call Groq chat completion with retry/backoff.
        Returns the assistant's text response.
        """
        time.sleep(MIN_REQUEST_INTERVAL)

        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content.strip()

            except AuthenticationError as e:
                raise ValueError(
                    "Groq API authentication failed (401/403). "
                    "Check your GROQ_API_KEY in .env."
                ) from e

            except RateLimitError as e:
                wait = _retry_wait_seconds(e, attempt)
                logger.warning(
                    "Rate limited by Groq. Waiting %.1fs (attempt %s/%s)",
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
                last_error = e

            except APIConnectionError as e:
                wait = BASE_BACKOFF * (attempt + 1)
                logger.warning(f"Connection error: {e}. Retrying in {wait}s...")
                time.sleep(wait)
                last_error = e

            except APIError as e:
                # Non-retryable API errors
                raise RuntimeError(f"Groq API error [{e.status_code}]: {e.message}") from e

            except Exception as e:
                raise RuntimeError(f"Unexpected LLM error: {e}") from e

        raise RuntimeError(
            f"Groq API failed after {MAX_RETRIES} retries. Last error: {last_error}"
        )

    def simple_prompt(self, prompt: str, system: Optional[str] = None, **kwargs) -> str:
        """Convenience wrapper for single-turn prompts."""
        return self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system,
            **kwargs,
        )


def _retry_wait_seconds(error: RateLimitError, attempt: int) -> float:
    retry_after = _retry_after_header(error)
    if retry_after is not None:
        return min(retry_after + 1.0, MAX_RETRY_WAIT)

    retry_from_message = _retry_after_message(str(error))
    if retry_from_message is not None:
        return min(retry_from_message + 1.0, MAX_RETRY_WAIT)

    return min(BASE_BACKOFF * (2 ** attempt), MAX_RETRY_WAIT)


def _retry_after_header(error: RateLimitError) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    value = headers.get("retry-after") or headers.get("Retry-After")
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _retry_after_message(message: str) -> Optional[float]:
    match = re.search(r"try again in (?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", message, re.I)
    if not match:
        return None

    minutes = float(match.group(1) or 0)
    seconds = float(match.group(2) or 0)
    return minutes * 60 + seconds
