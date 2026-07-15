from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from llm_factory import get_chat_llm, get_embeddings


KNOWLEDGE_BASE = [
    "50/30/20 budgeting splits after-tax income into 50% needs, 30% wants, and 20% savings or debt paydown.",
    "A savings rate is generally calculated as money left after expenses divided by income over the same period.",
    "Recurring subscriptions and dining spend are common areas for discretionary budget optimization.",
    "Fraud reviews should focus on unusual merchants, unusually large transactions, and abrupt changes in spending patterns.",
    "Investment suggestions in this app are educational and should not be treated as licensed financial advice.",
]


def _build_transaction_documents(transactions: pd.DataFrame, budgets: Optional[dict]) -> list[Document]:
    docs = [Document(page_content=text, metadata={"source": "glossary"}) for text in KNOWLEDGE_BASE]
    if transactions is None or transactions.empty:
        return docs

    spend_df = transactions[transactions["amount"] < 0].copy()
    if not spend_df.empty:
        spend_df["spend"] = spend_df["amount"].abs()
        category_summary = (
            spend_df.groupby("category")["spend"].sum().sort_values(ascending=False).round(2).to_dict()
        )
        merchant_summary = (
            spend_df.groupby("merchant")["spend"].sum().sort_values(ascending=False).head(10).round(2).to_dict()
        )
        docs.append(
            Document(
                page_content=f"User category spend summary: {category_summary}",
                metadata={"source": "user_summary"},
            )
        )
        docs.append(
            Document(
                page_content=f"User top merchants summary: {merchant_summary}",
                metadata={"source": "user_summary"},
            )
        )

    if budgets:
        docs.append(
            Document(
                page_content=f"Budget proposal summary: {budgets}",
                metadata={"source": "budget_summary"},
            )
        )
    return docs


def build_vectorstore(transactions: pd.DataFrame, budgets: Optional[dict] = None) -> FAISS:
    documents = _build_transaction_documents(transactions=transactions, budgets=budgets)
    embeddings = get_embeddings()
    return FAISS.from_documents(documents, embeddings)


def answer_finance_question(
    question: str,
    transactions: pd.DataFrame,
    budgets: Optional[dict],
    provider: str,
    api_key: Optional[str],
    top_k: int = 4,
) -> dict:
    vectorstore = build_vectorstore(transactions=transactions, budgets=budgets)
    retrieved_docs = vectorstore.similarity_search(question, k=top_k)
    context = "\n".join(f"- {doc.page_content}" for doc in retrieved_docs)

    try:
        llm = get_chat_llm(provider=provider, api_key=api_key)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "You are an AI personal finance advisor. Answer only using the supplied context "
                        "and clearly say when the data is insufficient."
                    )
                ),
                HumanMessage(
                    content=f"Context:\n{context}\n\nQuestion:\n{question}\n\nProvide a concise answer."
                ),
            ]
        )
        answer = response.content.strip()
    except Exception:
        answer = (
            "I could not reach the configured LLM provider, so here is the retrieved context instead:\n"
            f"{context}"
        )

    return {
        "answer": answer,
        "context_chunks": [doc.page_content for doc in retrieved_docs],
    }
