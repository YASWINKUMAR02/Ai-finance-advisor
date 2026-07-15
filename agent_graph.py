from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.budget_agent import run_budget_planning_agent
from agents.categorization_agent import run_expense_categorization_agent
from agents.fraud_agent import run_fraud_detection_agent
from agents.investment_agent import run_investment_recommendation_agent
from agents.state import AdvisorState
from agents.supervisor import route_intent


def _maybe_run_categorization(state: AdvisorState) -> AdvisorState:
    if "categorize" not in state.get("routed_agents", []):
        return {}
    return run_expense_categorization_agent(state)


def _maybe_run_budget(state: AdvisorState) -> AdvisorState:
    if "budget" not in state.get("routed_agents", []):
        return {}
    return run_budget_planning_agent(state)


def _maybe_run_investment(state: AdvisorState) -> AdvisorState:
    if "investment" not in state.get("routed_agents", []):
        return {}
    return run_investment_recommendation_agent(state)


def _maybe_run_fraud(state: AdvisorState) -> AdvisorState:
    if "fraud" not in state.get("routed_agents", []):
        return {}
    return run_fraud_detection_agent(state)


def build_graph():
    graph_builder = StateGraph(AdvisorState)
    graph_builder.add_node("supervisor", route_intent)
    graph_builder.add_node("categorize", _maybe_run_categorization)
    graph_builder.add_node("budget", _maybe_run_budget)
    graph_builder.add_node("investment", _maybe_run_investment)
    graph_builder.add_node("fraud", _maybe_run_fraud)

    graph_builder.add_edge(START, "supervisor")
    graph_builder.add_edge("supervisor", "categorize")
    graph_builder.add_edge("categorize", "budget")
    graph_builder.add_edge("budget", "investment")
    graph_builder.add_edge("investment", "fraud")
    graph_builder.add_edge("fraud", END)
    return graph_builder.compile()


ADVISOR_GRAPH = build_graph()


def run_advisor_graph(state: AdvisorState) -> AdvisorState:
    return ADVISOR_GRAPH.invoke(state)
