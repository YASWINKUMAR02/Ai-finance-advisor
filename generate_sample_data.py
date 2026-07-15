from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path


def main() -> None:
    random.seed(42)
    merchants = [
        ("Whole Foods", "Groceries", (-120, -35), ["groceries", "weekly grocery run"]),
        ("Trader Joes", "Groceries", (-95, -25), ["grocery refill", "produce and pantry"]),
        ("Starbucks", "Dining", (-18, -4), ["coffee", "breakfast coffee"]),
        ("Chipotle", "Dining", (-22, -8), ["lunch", "burrito bowl"]),
        ("Uber", "Transport", (-42, -9), ["ride share", "commute ride"]),
        ("Shell", "Transport", (-75, -28), ["fuel", "gas station"]),
        ("Landlord", "Housing", (-1650, -1450), ["rent", "monthly rent"]),
        ("Comcast", "Utilities", (-95, -70), ["internet bill", "home internet"]),
        ("Netflix", "Subscriptions", (-22, -15), ["streaming subscription", "monthly subscription"]),
        ("Spotify", "Subscriptions", (-14, -10), ["music subscription", "streaming music"]),
        ("Amazon", "Shopping", (-160, -18), ["online order", "household order"]),
        ("Target", "Shopping", (-140, -20), ["retail purchase", "household essentials"]),
        ("CVS Pharmacy", "Health", (-65, -12), ["pharmacy", "health purchase"]),
        ("Planet Fitness", "Health", (-30, -20), ["gym membership", "fitness"]),
        ("AMC", "Entertainment", (-45, -12), ["movie night", "cinema tickets"]),
        ("Delta", "Travel", (-480, -180), ["flight booking", "travel expense"]),
        ("Coursera", "Education", (-59, -29), ["course payment", "online course"]),
    ]

    rows = []
    start = date(2026, 1, 1)
    idx = 0

    while len(rows) < 108:
        current = start + timedelta(days=idx)
        idx += 1

        merchant_count = 1 if current.weekday() in (5, 6) else 2
        for merchant, category, bounds, descriptions in random.sample(merchants, k=merchant_count):
            if category == "Housing" and current.day != 1:
                continue
            if category in {"Utilities", "Subscriptions"} and current.day not in {3, 5, 10, 20}:
                continue
            if category == "Travel" and current.day not in {12, 26}:
                continue
            if category == "Education" and current.day not in {7, 21}:
                continue

            amount = round(random.uniform(min(bounds), max(bounds)), 2)
            rows.append([current.isoformat(), random.choice(descriptions), merchant, amount, category])
            if len(rows) >= 108:
                break

        if current.day in {1, 15} and len(rows) < 108:
            rows.append(
                [
                    current.isoformat(),
                    "salary deposit",
                    "Employer Payroll",
                    round(random.uniform(3600, 4300), 2),
                    "Income",
                ]
            )
            rows.append(
                [
                    current.isoformat(),
                    "transfer to savings",
                    "Marcus Savings",
                    round(random.uniform(-350, -150), 2),
                    "Savings",
                ]
            )

    output_path = Path("data/sample_transactions.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "description", "merchant", "amount", "category"])
        writer.writerows(rows[:108])

    print(f"Wrote {min(len(rows), 108)} rows to {output_path}")


if __name__ == "__main__":
    main()
