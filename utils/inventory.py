import numpy as np
import pandas as pd


def compute_reorder_point(forecast: pd.Series, lead_time_days: int, safety_z: float) -> float:
    mean_daily = forecast.mean()
    std_daily = forecast.std()
    safety_stock = safety_z * std_daily * np.sqrt(lead_time_days)
    return mean_daily * lead_time_days + safety_stock


def simulate_inventory_policy(actual: pd.Series, reorder_point: float, initial_inventory: float) -> pd.DataFrame:
    inventory = initial_inventory
    records = []
    for date, demand in actual.items():
        inventory -= demand
        if inventory < reorder_point:
            # Simple replenishment: bring inventory back to 2x reorder point.
            inventory = reorder_point * 2
        records.append({"date": date, "inventory_level": max(inventory, 0), "demand": demand})
    return pd.DataFrame(records).set_index("date")


def inventory_mismatch_cost(sim_df: pd.DataFrame) -> float:
    overstock = (sim_df["inventory_level"] - sim_df["demand"]).clip(lower=0).sum()
    stockout = (sim_df["demand"] - sim_df["inventory_level"]).clip(lower=0).sum()
    # Stockout is more expensive.
    return overstock * 1.0 + stockout * 2.0
