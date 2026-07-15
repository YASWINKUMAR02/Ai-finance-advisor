from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain_core.messages import HumanMessage

from agents.state import AdvisorState
from expense_classifier import predict_categories
from llm_factory import get_chat_llm


def _llm_category_fallback(
    description: str,
    merchant: str,
    provider: str,
    api_key: Optional[str],
) -> str:
    try:
        llm = get_chat_llm(provider=provider, api_key=api_key)
    except Exception:
        return "Other"

    prompt = (
        "Classify this personal finance transaction into one concise category. "
        "Use one of: Groceries, Dining, Transport, Housing, Utilities, Shopping, "
        "Health, Entertainment, Travel, Income, Savings, Subscriptions, Education, Other. "
        f"Merchant: {merchant}\nDescription: {description}\nCategory:"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip().splitlines()[0][:40] or "Other"


def run_expense_categorization_agent(state: AdvisorState) -> AdvisorState:
    transactions = state.get("transactions_df")
    if transactions is None or transactions.empty:
        return {"categorized_df": pd.DataFrame()}

    predictions = predict_categories(transactions)
    provider = state.get("provider", "openai")
    api_key = state.get("api_key")

    def assign_category(row: pd.Series) -> str:
        existing = row.get("category")
        if pd.notna(existing) and str(existing).strip():
            return str(existing)
        if not row["needs_llm_review"]:
            return str(row["predicted_category"])
        return _llm_category_fallback(
            description=str(row.get("description", "")),
            merchant=str(row.get("merchant", "")),
            provider=provider,
            api_key=api_key,
        )

    predictions["final_category"] = predictions.apply(assign_category, axis=1)
    predictions["category"] = predictions["final_category"]

    category_spend = (
        predictions[predictions["amount"] < 0]
        .assign(spend=lambda frame: frame["amount"].abs())
        .groupby("category")["spend"]
        .sum()
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "categorized_df": predictions.drop(columns=["final_category"]),
        "category_metrics": {"spend_by_category": category_spend},
    }
