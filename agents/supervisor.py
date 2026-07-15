from __future__ import annotations

from agents.state import AdvisorState


ROUTE_KEYWORDS = {
    "categorize": ["categorize", "category", "classify", "merchant"],
    "budget": ["budget", "save", "spending plan", "overspend"],
    "investment": ["invest", "allocation", "portfolio", "risk"],
    "fraud": ["fraud", "suspicious", "anomaly", "odd transaction"],
}


def route_intent(state: AdvisorState) -> AdvisorState:
    query = state.get("user_query", "").lower()
    selected = []

    for route, keywords in ROUTE_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            selected.append(route)

    if not selected:
        selected = ["categorize", "budget", "investment", "fraud"]
    elif "categorize" not in selected and any(
        route in selected for route in ["budget", "investment", "fraud"]
    ):
        selected.insert(0, "categorize")

    return {
        "routed_agents": selected,
        "next_agent": selected[0],
    }
