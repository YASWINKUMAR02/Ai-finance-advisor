# AI Personal Finance Advisor

An agentic personal finance project built with Python, LangGraph, LangChain, FAISS, scikit-learn, statsmodels, and Streamlit.

It accepts transaction CSV or Excel uploads, classifies spending categories, forecasts future spend, proposes budgets, flags suspicious transactions, and provides a grounded chat assistant using retrieval-augmented generation.

## Features

- CSV and Excel upload pipeline with date normalization and deduplication
- Synthetic sample dataset in `data/sample_transactions.csv`
- Expense classification model using TF-IDF + Logistic Regression
- Saved model artifact in `models/expense_classifier.joblib`
- Spend forecasting by category using statsmodels SARIMAX with a lightweight fallback for short histories
- LangGraph supervisor plus specialized agents for categorization, budgets, investments, and fraud checks
- FAISS-backed RAG assistant over budgeting concepts and user-specific spend summaries
- Streamlit dashboard with charts, budget tracking, fraud alerts, and chat UI

## Project Structure

```text
.
|-- app.py
|-- agent_graph.py
|-- data_loader.py
|-- expense_classifier.py
|-- forecaster.py
|-- rag_engine.py
|-- llm_factory.py
|-- train_classifier.py
|-- generate_sample_data.py
|-- requirements.txt
|-- .env.example
|-- agents/
|   |-- state.py
|   |-- supervisor.py
|   |-- categorization_agent.py
|   |-- budget_agent.py
|   |-- investment_agent.py
|   `-- fraud_agent.py
|-- data/
|   `-- sample_transactions.csv
`-- models/
    |-- expense_classifier.joblib
    `-- classifier_metrics.json
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy the environment template and add your API keys if you want LLM-backed categorization or chat generation:

```bash
copy .env.example .env
```

4. Optionally regenerate the bundled sample data and retrain the classifier artifact:

```bash
python generate_sample_data.py
python train_classifier.py
```

## Run The App

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

If you do not upload a file, the app uses `data/sample_transactions.csv` by default.

## Supported Input Formats

- `.csv`
- `.xlsx`

## Expected Input Schema

Required columns:

- `date`
- `description`
- `merchant`
- `amount`

Optional column:

- `category`

Expense amounts are expected to be negative and income amounts positive.

## Architecture Overview

### 1. Data Layer

- `data_loader.py` loads CSV and Excel files with pandas
- normalizes column names and dates
- trims text fields and drops malformed rows
- deduplicates transactions on `date`, `description`, `merchant`, and `amount`

### 2. ML Layer

- `expense_classifier.py` trains and serves a TF-IDF + Logistic Regression pipeline
- `train_classifier.py` saves the trained classifier to `models/expense_classifier.joblib`
- `forecaster.py` aggregates historical spend and forecasts future category spend with SARIMAX

### 3. Multi-Agent Layer

`agent_graph.py` compiles a LangGraph `StateGraph` over a shared `AdvisorState` TypedDict.

Agents:

- `supervisor.py`: routes the user intent to the appropriate agents
- `categorization_agent.py`: predicts categories with the ML model and falls back to an LLM for low-confidence cases
- `budget_agent.py`: combines actual spend and forecasted spend to propose a category-level budget using a 50/30/20 baseline
- `investment_agent.py`: generates educational asset allocation suggestions based on savings rate and risk tolerance
- `fraud_agent.py`: flags suspicious transactions using z-score, isolation forest, and new-merchant heuristics

### 4. RAG Layer

- `rag_engine.py` builds a FAISS vector store from:
  - budgeting glossary snippets
  - user category spend summaries
  - user merchant summaries
  - budget summaries
- the app retrieves top-k relevant chunks and passes them to an LLM
- LLM provider is selected via the UI and uses API keys loaded from `.env` or environment variables
- embeddings are local hash embeddings so retrieval still works when no external embeddings API is configured

### 5. Streamlit Layer

`app.py` provides:

- sidebar controls for CSV or Excel upload, risk tolerance, and provider selection
- `Overview` tab for category charts and monthly trends
- `Budget` tab for proposed budget versus actual spend
- `Investments` tab for educational allocation suggestions
- `Fraud alerts` tab for suspicious transaction review
- `Ask the advisor` tab with `st.chat_message` and grounded RAG responses

## Notes

- OpenAI or Groq API keys are optional for basic local analysis and are read from `.env` or environment variables.
- Without an API key, the app still loads data, runs the ML/forecast/fraud pipeline, and returns retrieved context in the chat fallback path.
- The included sample classifier metrics are stored in `models/classifier_metrics.json`.
