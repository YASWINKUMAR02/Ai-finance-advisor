from __future__ import annotations

from typing import Any, Optional, TypedDict

import pandas as pd


class AdvisorState(TypedDict, total=False):
    user_query: str
    transactions_df: pd.DataFrame
    categorized_df: pd.DataFrame
    forecasts: dict[str, Any]
    budgets: dict[str, Any]
    fraud_flags: pd.DataFrame
    investment_profile: dict[str, Any]
    investment_recommendation: dict[str, Any]
    risk_tolerance: str
    routed_agents: list[str]
    next_agent: str
    provider: str
    api_key: Optional[str]
    category_metrics: dict[str, Any]
    chat_history: list[dict[str, str]]
    rag_context: list[str]
    assistant_response: str
