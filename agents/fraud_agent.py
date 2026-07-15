from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from agents.state import AdvisorState


def run_fraud_detection_agent(state: AdvisorState) -> AdvisorState:
    transactions = state.get("categorized_df")
    if transactions is None:
        transactions = state.get("transactions_df")
    if transactions is None or transactions.empty:
        return {"fraud_flags": pd.DataFrame()}

    working = transactions.copy()
    spend = working[working["amount"] < 0].copy()
    if spend.empty:
        return {"fraud_flags": pd.DataFrame()}

    spend["abs_amount"] = spend["amount"].abs()
    mean_amount = spend["abs_amount"].mean()
    std_amount = spend["abs_amount"].std(ddof=0) or 1.0
    spend["amount_zscore"] = (spend["abs_amount"] - mean_amount) / std_amount

    model = IsolationForest(contamination=0.08, random_state=42)
    spend["isolation_flag"] = model.fit_predict(spend[["abs_amount"]])

    merchant_counts = spend["merchant"].value_counts()
    spend["new_merchant_flag"] = spend["merchant"].map(merchant_counts).fillna(0) == 1

    reasons = []
    for _, row in spend.iterrows():
        row_reasons = []
        if row["amount_zscore"] > 2.5:
            row_reasons.append("Large amount compared with typical spend")
        if row["isolation_flag"] == -1:
            row_reasons.append("Isolation forest anomaly")
        if row["new_merchant_flag"] and row["abs_amount"] > spend["abs_amount"].median() * 1.5:
            row_reasons.append("High-value purchase from a new merchant")
        reasons.append("; ".join(row_reasons))

    spend["reason"] = reasons
    flagged = spend[spend["reason"] != ""].copy()
    if flagged.empty:
        flagged = spend.nlargest(min(3, len(spend)), "amount_zscore").copy()
        flagged["reason"] = "Top statistical outlier by amount"

    output_columns = ["date", "merchant", "description", "amount", "category", "reason"]
    for column in output_columns:
        if column not in flagged.columns:
            flagged[column] = np.nan

    return {
        "fraud_flags": flagged[output_columns].sort_values("date", ascending=False).reset_index(drop=True)
    }
