import numpy as np
import pandas as pd

from .config import SEED, N_PRODUCTS, START_DATE, END_DATE


def generate_synthetic_data() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    n_days = len(dates)

    records = []
    for i in range(N_PRODUCTS):
        product_id = f"P{i+1:03d}"
        base = rng.integers(20, 120)
        trend = rng.uniform(0.01, 0.05)
        weekly = 8 * np.sin(2 * np.pi * np.arange(n_days) / 7)
        yearly = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        noise = rng.normal(0, 6, n_days)
        demand = np.maximum(0, base + trend * np.arange(n_days) + weekly + yearly + noise)

        # Inventory level is a noisy function of demand with a basic buffer.
        buffer = rng.integers(50, 120)
        inventory = np.maximum(0, demand + buffer + rng.normal(0, 10, n_days))

        for d, s, inv in zip(dates, demand, inventory):
            records.append(
                {
                    "date": d,
                    "product_id": product_id,
                    "daily_sales": round(float(s), 2),
                    "inventory_level": round(float(inv), 2),
                }
            )

    df = pd.DataFrame(records)

    # Inject missing values and outliers for preprocessing steps.
    missing_idx = df.sample(frac=0.01, random_state=SEED).index
    df.loc[missing_idx, "daily_sales"] = np.nan
    outlier_idx = df.sample(frac=0.005, random_state=SEED + 1).index
    df.loc[outlier_idx, "daily_sales"] *= 4

    return df
