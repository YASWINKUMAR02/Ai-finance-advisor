from __future__ import annotations

from agents.state import AdvisorState


RISK_ALLOCATIONS = {
    "Conservative": {"Cash": 0.2, "Bonds": 0.55, "Index Funds": 0.2, "International": 0.05},
    "Balanced": {"Cash": 0.1, "Bonds": 0.3, "Index Funds": 0.45, "International": 0.15},
    "Aggressive": {"Cash": 0.05, "Bonds": 0.15, "Index Funds": 0.6, "International": 0.2},
}


def run_investment_recommendation_agent(state: AdvisorState) -> AdvisorState:
    categorized = state.get("categorized_df")
    risk_tolerance = state.get("risk_tolerance", "Balanced").title()

    if categorized is None or categorized.empty:
        return {
            "investment_recommendation": {
                "allocation": RISK_ALLOCATIONS.get(risk_tolerance, RISK_ALLOCATIONS["Balanced"]),
                "savings_rate": 0.0,
                "disclaimer": "Educational only. This is not licensed financial advice.",
                "rationale": "Upload transactions to personalize the recommendation.",
            }
        }

    income = float(categorized.loc[categorized["amount"] > 0, "amount"].sum())
    spend = float(categorized.loc[categorized["amount"] < 0, "amount"].abs().sum())
    savings_rate = max((income - spend) / income, 0.0) if income > 0 else 0.0

    allocation = RISK_ALLOCATIONS.get(risk_tolerance, RISK_ALLOCATIONS["Balanced"]).copy()
    if savings_rate < 0.1:
        allocation["Cash"] = round(allocation["Cash"] + 0.05, 2)
        allocation["Index Funds"] = round(max(allocation["Index Funds"] - 0.05, 0.0), 2)

    rationale = (
        f"Estimated savings rate is {savings_rate:.1%}. The allocation emphasizes "
        f"{risk_tolerance.lower()} risk capacity while keeping a cash buffer that fits current savings."
    )

    return {
        "investment_profile": {
            "risk_tolerance": risk_tolerance,
            "income": round(income, 2),
            "spend": round(spend, 2),
            "savings_rate": round(savings_rate, 4),
        },
        "investment_recommendation": {
            "allocation": allocation,
            "savings_rate": round(savings_rate, 4),
            "disclaimer": "Educational only. This is not licensed financial advice.",
            "rationale": rationale,
        },
    }
