from __future__ import annotations
import os

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agent_graph import run_advisor_graph
from data_loader import load_transactions
from rag_engine import answer_finance_question


load_dotenv()
st.set_page_config(page_title="AI Personal Finance Advisor", layout="wide")


def _load_dataframe(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return load_transactions(uploaded_file)
    return load_transactions("data/sample_transactions.csv")


def _run_analysis(df: pd.DataFrame, provider: str, risk_tolerance: str, api_key: str) -> dict:
    initial_state = {
        "user_query": "categorize budget investment fraud",
        "transactions_df": df,
        "provider": provider,
        "api_key": api_key,
        "risk_tolerance": risk_tolerance,
        "chat_history": st.session_state.get("chat_history", []),
    }
    return run_advisor_graph(initial_state)


def _category_spend_frame(categorized_df: pd.DataFrame) -> pd.DataFrame:
    spend_df = categorized_df[categorized_df["amount"] < 0].copy()
    spend_df["spend"] = spend_df["amount"].abs()
    return (
        spend_df.groupby("category")["spend"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )


def _monthly_trend_frame(categorized_df: pd.DataFrame) -> pd.DataFrame:
    spend_df = categorized_df[categorized_df["amount"] < 0].copy()
    spend_df["spend"] = spend_df["amount"].abs()
    trend = (
        spend_df.groupby([pd.Grouper(key="date", freq="ME"), "category"])["spend"]
        .sum()
        .reset_index()
    )
    trend["month"] = trend["date"].dt.strftime("%Y-%m")
    return trend


def _budget_comparison_frame(categorized_df: pd.DataFrame, budgets: dict) -> pd.DataFrame:
    actual_df = _category_spend_frame(categorized_df).rename(columns={"spend": "actual_spend"})
    proposed = budgets.get("proposed_by_category", {})
    actual_df["proposed_budget"] = actual_df["category"].map(proposed).fillna(0.0)
    actual_df["progress_ratio"] = (
        actual_df["actual_spend"] / actual_df["proposed_budget"].replace(0, pd.NA)
    ).fillna(0.0)
    return actual_df.sort_values("actual_spend", ascending=False)


st.title("AI Personal Finance Advisor")
st.caption("Agentic finance workflow built with LangGraph, LangChain, FAISS, and Streamlit.")

default_provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
provider_options = ["openai", "groq"]
default_provider_index = provider_options.index(default_provider) if default_provider in provider_options else 0

with st.sidebar:
    st.header("Configuration")
    st.caption("Required columns: `date`, `description`, `merchant`, `amount`")
    st.caption("*(Note: Use negative amounts for expenditure and positive for salary/income)*")
    uploaded_file = st.file_uploader("Upload transactions file", type=["csv", "xlsx"])
    risk_tolerance = st.selectbox("Risk tolerance", ["Conservative", "Balanced", "Aggressive"], index=1)
    provider = st.selectbox("LLM provider", provider_options, index=default_provider_index)
    
    api_key = st.text_input(
        f"{provider.capitalize()} API Key",
        value="",
        type="password",
        placeholder=f"Enter {provider} key..."
    )

    use_sample = uploaded_file is None
    st.info("Using bundled sample data." if use_sample else "Using uploaded transaction data.")
    st.caption("Supported upload formats: `.csv`, `.xlsx`")
    st.caption("API keys can also be loaded from the `.env` file or environment variables.")

provider_has_key = bool(api_key.strip())

transactions_df = _load_dataframe(uploaded_file)
analysis = _run_analysis(transactions_df, provider=provider, risk_tolerance=risk_tolerance, api_key=api_key)
categorized_df = analysis.get("categorized_df", transactions_df)
budgets = analysis.get("budgets", {})
investment = analysis.get("investment_recommendation", {})
fraud_flags = analysis.get("fraud_flags", pd.DataFrame())

overview_tab, budget_tab, invest_tab, fraud_tab, ask_tab = st.tabs(
    ["Overview", "Budget", "Investments", "Fraud alerts", "Ask the advisor"]
)

with overview_tab:
    st.subheader("Spending Overview")
    col1, col2 = st.columns(2)
    category_spend = _category_spend_frame(categorized_df)
    monthly_trend = _monthly_trend_frame(categorized_df)

    with col1:
        if not category_spend.empty:
            bar_chart = px.bar(category_spend, x="category", y="spend", title="Spend by Category")
            st.plotly_chart(bar_chart, use_container_width=True)
        else:
            st.info("No expense data available for Category Spend chart.")
            
    with col2:
        if not category_spend.empty:
            pie_chart = px.pie(category_spend, names="category", values="spend", title="Category Mix")
            st.plotly_chart(pie_chart, use_container_width=True)
        else:
            st.info("No expense data available for Category Mix chart.")

    if not monthly_trend.empty:
        if monthly_trend["month"].nunique() == 1:
            trend_chart = px.bar(
                monthly_trend,
                x="month",
                y="spend",
                color="category",
                barmode="group",
                title="Monthly Spend Trend (Single Month)",
            )
        else:
            trend_chart = px.line(
                monthly_trend,
                x="month",
                y="spend",
                color="category",
                markers=True,
                title="Monthly Spend Trend",
            )
        st.plotly_chart(trend_chart, use_container_width=True)
    else:
        st.info("No expense data available for Monthly Trend chart.")
        
    st.dataframe(categorized_df[["date", "merchant", "description", "amount", "category"]], use_container_width=True)

with budget_tab:
    st.subheader("Budget Plan")
    budget_frame = _budget_comparison_frame(categorized_df, budgets)
    st.write(budgets.get("explanation", "No budget recommendation available yet."))
    st.dataframe(budget_frame[["category", "actual_spend", "proposed_budget"]], use_container_width=True)

    for _, row in budget_frame.iterrows():
        progress = min(float(row["progress_ratio"]), 1.0)
        st.write(
            f"{row['category']}: actual ${row['actual_spend']:.2f} vs budget ${row['proposed_budget']:.2f}"
        )
        st.progress(progress, text=f"{row['progress_ratio']:.0%} of proposed budget used")

with invest_tab:
    st.subheader("Investment Suggestions")
    st.write(investment.get("rationale", "No investment recommendation available yet."))
    st.warning(investment.get("disclaimer", "Educational only."))

    allocation = investment.get("allocation", {})
    allocation_df = pd.DataFrame(
        {"Asset class": list(allocation.keys()), "Allocation": [value * 100 for value in allocation.values()]}
    )
    if not allocation_df.empty:
        allocation_chart = px.bar(allocation_df, x="Asset class", y="Allocation", title="Suggested Allocation (%)")
        st.plotly_chart(allocation_chart, use_container_width=True)

with fraud_tab:
    st.subheader("Fraud Alerts")
    if fraud_flags.empty:
        st.success("No suspicious transactions were flagged.")
    else:
        st.dataframe(fraud_flags, use_container_width=True)

with ask_tab:
    st.subheader("Ask the Advisor")
    if not provider_has_key:
        st.info(
            f"No API key found for `{provider}`. Add it to `.env` to enable live LLM answers. "
            "The app will still use retrieval fallback."
        )
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask about your spending, budgets, or trends")
    if prompt:
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        rag_result = answer_finance_question(
            question=prompt,
            transactions=categorized_df,
            budgets=budgets,
            provider=provider,
            api_key=api_key,
        )
        answer = rag_result["answer"]
        st.session_state["chat_history"].append({"role": "assistant", "content": answer})

        with st.chat_message("assistant"):
            st.markdown(answer)
            with st.expander("Retrieved context"):
                for chunk in rag_result["context_chunks"]:
                    st.write(chunk)
