from __future__ import annotations

from typing import Optional

import pandas as pd
from langchain_core.messages import HumanMessage

from agents.state import AdvisorState
from forecaster import forecast_category_spend, forecast_to_frame
from llm_factory import get_chat_llm


NEEDS_BUCKETS = {
    "Housing": "Needs",
    "Utilities": "Needs",
    "Groceries": "Needs",
    "Transport": "Needs",
    "Health": "Needs",
    "Education": "Needs",
    "Dining": "Wants",
    "Entertainment": "Wants",
    "Shopping": "Wants",
    "Travel": "Wants",
    "Subscriptions": "Wants",
    "Savings": "Savings",
    "Income": "Income",
    "Other": "Wants",
}


def _generate_budget_narrative(
    summary: dict,
    provider: str,
    api_key: Optional[str],
) -> str:
    try:
        llm = get_chat_llm(provider=provider, api_key=api_key)
    except Exception:
        return (
            "Budget uses a 50/30/20 baseline adjusted toward the categories with the "
            "largest forecasted spend and current overspend."
        )

    prompt = (
        "You are a personal finance assistant. Using the data below, write a brief budget "
        "recommendation in 4 sentences or fewer. Mention the 50/30/20 baseline and one or "
        "two practical refinements.\n"
        f"{summary}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def run_budget_planning_agent(state: AdvisorState) -> AdvisorState:
    categorized = state.get("categorized_df")
    if categorized is None or categorized.empty:
        return {"budgets": {}}

    forecasts = state.get("forecasts")
    if not forecasts:
        forecasts = forecast_category_spend(categorized)

    spend_df = categorized[categorized["amount"] < 0].copy()
    spend_df["spend"] = spend_df["amount"].abs()
    actuals = spend_df.groupby("category")["spend"].sum().sort_values(ascending=False)

    income_total = float(categorized.loc[categorized["amount"] > 0, "amount"].sum())
    total_spend = float(spend_df["spend"].sum())
    if income_total <= 0:
        income_total = total_spend / 0.8 if total_spend else 1.0

    target_needs = income_total * 0.5
    target_wants = income_total * 0.3
    target_savings = income_total * 0.2

    forecast_frame = forecast_to_frame(forecasts)
    future_spend = (
        forecast_frame.groupby("category")["forecast_spend"].mean().to_dict()
        if not forecast_frame.empty
        else {}
    )

    proposed = {}
    for category, actual in actuals.items():
        bucket = NEEDS_BUCKETS.get(category, "Wants")
        bucket_target = {
            "Needs": target_needs,
            "Wants": target_wants,
            "Savings": target_savings,
            "Income": income_total,
        }[bucket]
        peer_categories = [c for c, b in NEEDS_BUCKETS.items() if b == bucket and c in actuals.index]
        divisor = max(len(peer_categories), 1)
        baseline = bucket_target / divisor
        projected = future_spend.get(category, actual / max(len(spend_df["date"].dt.to_period("M").unique()), 1))
        proposed[category] = round(max(baseline * 0.8, projected), 2)

    summary = {
        "income_total": round(income_total, 2),
        "total_spend": round(total_spend, 2),
        "actuals": actuals.round(2).to_dict(),
        "forecast_avg": future_spend,
        "proposed_budget": proposed,
    }

    explanation = _generate_budget_narrative(
        summary=summary,
        provider=state.get("provider", "openai"),
        api_key=state.get("api_key"),
    )

    return {
        "forecasts": forecasts,
        "budgets": {
            "income_total": round(income_total, 2),
            "actual_spend_total": round(total_spend, 2),
            "proposed_by_category": proposed,
            "framework_targets": {
                "Needs": round(target_needs, 2),
                "Wants": round(target_wants, 2),
                "Savings": round(target_savings, 2),
            },
            "explanation": explanation,
        },
    }
