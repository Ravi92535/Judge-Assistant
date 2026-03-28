from __future__ import annotations
"""
provider.py — Centralized LLM provider for Judge Assistant.

Uses Groq (llama-3.1-8b-instant) as the single, fast inference backend.
The public surface is get_llm_response(), which every core module calls.
"""

import os
from dotenv import load_dotenv
from groq import Groq
from core.utils import retry_on_quota

# Load .env from project root (two levels up from this file).
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


class LLMProvider:
    """
    Thin wrapper around the Groq client.

    Args:
        provider: Reserved for future multi-provider support. Currently only
                  "groq" is used.
    """

    def __init__(self, provider: str = "groq"):
        self.provider = provider
        self.groq_client: Groq | None = None
        if GROQ_API_KEY:
            self.groq_client = Groq(api_key=GROQ_API_KEY)
        else:
            print("WARNING: GROQ_API_KEY not set. LLM calls will fail.")

    @retry_on_quota
    def generate_content(
        self,
        prompt: str,
        is_json: bool = False,
        allow_fallback: bool = True,  # BUG FIX: documented — reserved for future provider fallback
    ) -> str:
        """
        Generate a response from the LLM.

        Args:
            prompt: The user prompt to send.
            is_json: When True, instructs the model to return a JSON object
                     and sets response_format accordingly.
            allow_fallback: Currently unused (Groq-only). Reserved so that
                            callers don't need to be changed when a secondary
                            provider is added in the future.

        Returns:
            The model's response as a plain string.

        Raises:
            RuntimeError: If no API key is configured.
            groq.RateLimitError: Propagated and handled by @retry_on_quota.
        """
        if not self.groq_client:
            raise RuntimeError(
                "Groq API key is missing. Set GROQ_API_KEY in your .env file."
            )

        response_format = {"type": "json_object"} if is_json else None

        completion = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional legal assistant. "
                        "Output only the requested format."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            response_format=response_format,
        )
        return completion.choices[0].message.content


# ─── Module-level singleton ───────────────────────────────────────────────────
_provider = LLMProvider(provider="groq")


def get_llm_response(
    prompt: str,
    is_json: bool = False,
    allow_fallback: bool = True,
) -> str:
    """
    Public helper — get an LLM response from the shared provider instance.

    Args:
        prompt: The prompt to send to the model.
        is_json: Whether the model should return a JSON object.
        allow_fallback: Reserved for future multi-provider support.

    Returns:
        The model's response as a string.
    """
    return _provider.generate_content(
        prompt, is_json=is_json, allow_fallback=allow_fallback
    )


if __name__ == "__main__":
    print(get_llm_response("What is the capital of France?"))
