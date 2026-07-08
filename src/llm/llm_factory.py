import os
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from ..enums import LLMProvider


class LLMFactory:
    """
    Resolves a LangChain BaseChatModel for the configured provider.

    Returning a plain BaseChatModel (rather than a custom wrapper) is the
    whole point of the LangChain migration: every downstream chain is
    built with LCEL (`prompt | llm.with_structured_output(Schema)`), which
    works identically regardless of which concrete chat model this
    factory hands back.
    """

    @staticmethod
    def create(
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> BaseChatModel:
        provider = provider or LLMProvider(os.environ.get("LLM_PROVIDER", LLMProvider.GROQ.value))

        if provider == LLMProvider.GROQ:
            from langchain_groq import ChatGroq

            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY is not set.")

            return ChatGroq(
                model=model or os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
                temperature=temperature,
                api_key=api_key,
            )

        if provider == LLMProvider.GEMINI:
            from langchain_google_genai import ChatGoogleGenerativeAI

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set.")

            return ChatGoogleGenerativeAI(
                model=model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
                temperature=temperature,
                google_api_key=api_key,
            )

        raise ValueError(f"Unsupported LLM provider: {provider}")
