import os
from typing import Optional

from langchain_core.embeddings import Embeddings


class EmbeddingFactory:
    """
    Resolves a LangChain Embeddings implementation. Everything downstream
    (Chroma via langchain_chroma) speaks the LangChain Embeddings
    interface natively, so there's no need for a custom EmbeddingModel
    ABC anymore -- this factory just picks which concrete implementation
    to hand back.
    """

    @staticmethod
    def create(
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> Embeddings:
        provider = (provider or os.environ.get("EMBEDDING_PROVIDER", "huggingface")).lower()

        if provider == "huggingface":
            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=model_name or os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            )

        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings

            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set.")

            return OpenAIEmbeddings(
                model=model_name or os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
                api_key=api_key,
            )

        raise ValueError(f"Unsupported embedding provider: {provider}")
