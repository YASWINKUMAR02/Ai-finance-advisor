from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


@dataclass
class ForecastResult:
    category: str
    history: pd.DataFrame
    forecast: pd.DataFrame


def aggregate_spend(
    df: pd.DataFrame,
    frequency: str = "ME",
    category_column: str = "category",
) -> pd.DataFrame:
    working = df.copy()
    working = working[working["amount"] < 0]
    working["spend"] = working["amount"].abs()
    grouped = (
        working.groupby([pd.Grouper(key="date", freq=frequency), category_column])["spend"]
        .sum()
        .reset_index()
    )
    return grouped.sort_values(["date", category_column]).reset_index(drop=True)


def forecast_category_spend(
    df: pd.DataFrame,
    periods: int = 3,
    frequency: str = "ME",
    category_column: str = "category",
) -> dict[str, ForecastResult]:
    aggregated = aggregate_spend(df, frequency=frequency, category_column=category_column)
    results: dict[str, ForecastResult] = {}

    for category, group in aggregated.groupby(category_column):
        indexed = group.set_index("date")["spend"].asfreq(frequency, fill_value=0.0)
        if len(indexed) < 6 or indexed.nunique() < 2:
            future_dates = pd.date_range(
                indexed.index.max() if len(indexed) else pd.Timestamp.today().normalize(),
                periods=periods + 1,
                freq=frequency,
            )[1:]
            forecast_values = pd.Series([float(indexed.mean() if len(indexed) else 0.0)] * periods)
        else:
            model = SARIMAX(
                indexed,
                order=(1, 1, 1),
                seasonal_order=(0, 0, 0, 0),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            fitted = model.fit(disp=False)
            prediction = fitted.get_forecast(steps=periods)
            future_dates = prediction.predicted_mean.index
            forecast_values = prediction.predicted_mean.clip(lower=0)

        forecast_df = pd.DataFrame(
            {
                "date": future_dates,
                "forecast_spend": [round(float(value), 2) for value in forecast_values],
            }
        )
        history_df = indexed.reset_index().rename(columns={"spend": "historical_spend"})
        results[str(category)] = ForecastResult(
            category=str(category),
            history=history_df,
            forecast=forecast_df,
        )

    return results


def forecast_to_frame(results: dict[str, ForecastResult]) -> pd.DataFrame:
    frames = []
    for category, result in results.items():
        forecast = result.forecast.copy()
        forecast["category"] = category
        frames.append(forecast)

    if not frames:
        return pd.DataFrame(columns=["date", "forecast_spend", "category"])
    return pd.concat(frames, ignore_index=True)
