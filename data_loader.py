from __future__ import annotations

from pathlib import Path
from typing import BinaryIO, Union

import pandas as pd


REQUIRED_COLUMNS = ["date", "description", "merchant", "amount"]
OPTIONAL_COLUMNS = ["category"]
ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS


def _normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working.columns = [str(column).strip().lower() for column in working.columns]

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in working.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in REQUIRED_COLUMNS:
        if column not in working.columns:
            working[column] = None

    if "category" not in working.columns:
        working["category"] = None

    working = working[ALL_COLUMNS]
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", "amount"])

    working["description"] = _normalize_text(working["description"])
    working["merchant"] = _normalize_text(working["merchant"])
    working["category"] = (
        working["category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
    )
    working["amount"] = pd.to_numeric(working["amount"], errors="coerce")
    working = working.dropna(subset=["amount"])

    working["amount"] = working["amount"].round(2)
    working["date"] = working["date"].dt.normalize()
    working = working.drop_duplicates(
        subset=["date", "description", "merchant", "amount"],
        keep="first",
    )
    return working.sort_values("date").reset_index(drop=True)


def load_transactions(source: Union[str, Path, BinaryIO]) -> pd.DataFrame:
    source_name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "")
    suffix = Path(source_name).suffix.lower()

    if suffix == ".xlsx":
        dataframe = pd.read_excel(source, engine="openpyxl")
    else:
        dataframe = pd.read_csv(source)
    return clean_transactions(dataframe)


def summarize_transactions(df: pd.DataFrame) -> dict:
    cleaned = clean_transactions(df)
    spend = cleaned[cleaned["amount"] < 0].copy()
    spend["abs_amount"] = spend["amount"].abs()

    summary = {
        "row_count": int(len(cleaned)),
        "date_min": cleaned["date"].min(),
        "date_max": cleaned["date"].max(),
        "total_spend": float(spend["abs_amount"].sum()),
        "total_income": float(cleaned.loc[cleaned["amount"] > 0, "amount"].sum()),
        "category_spend": spend.groupby("category", dropna=False)["abs_amount"].sum().to_dict(),
    }
    return summary
