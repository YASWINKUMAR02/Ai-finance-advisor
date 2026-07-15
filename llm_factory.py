from __future__ import annotations

import os
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import HashingVectorizer


class LocalHashEmbeddings(Embeddings):
    def __init__(self, n_features: int = 256) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> list[float]:
        matrix = self.vectorizer.transform([text])
        return matrix.toarray()[0].tolist()


def get_chat_llm(provider: str, api_key: Optional[str] = None) -> BaseChatModel:
    normalized = provider.lower().strip()
    if normalized == "groq":
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Missing Groq API key.")
        return ChatGroq(
            groq_api_key=key,
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            temperature=0.2,
        )

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Missing OpenAI API key.")
    return ChatOpenAI(
        api_key=key,
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.2,
    )


def get_embeddings() -> Embeddings:
    return LocalHashEmbeddings()
