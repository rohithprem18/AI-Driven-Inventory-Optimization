import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.arima_model import forecast_arima, train_arima, tune_arima
from models.lstm_model import build_lstm, forecast_lstm, train_lstm
from utils.config import (
    LEAD_TIME_DAYS,
    LSTM_BATCH,
    LSTM_EPOCHS,
    LSTM_WINDOW,
    SERVICE_LEVEL_Z,
    TEST_DAYS,
)
from utils.data_generation import generate_synthetic_data
from utils.inventory import compute_reorder_point, inventory_mismatch_cost, simulate_inventory_policy
from utils.preprocess import (
    adf_test,
    handle_missing_and_outliers,
    make_lstm_sequences,
    scale_for_lstm,
    train_test_split_by_time,
)
from visualizations.plots import plot_actual_vs_pred, plot_inventory_levels, plot_sales_trends


def ensure_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("visualizations", exist_ok=True)


def main():
    ensure_dirs()

    # 1) Data Design
    df_raw = generate_synthetic_data()
    df_raw.to_csv("data/synthetic_sales.csv", index=False)

    # 2) Preprocessing
    df = handle_missing_and_outliers(df_raw)

    # Focus on a single product for modeling (easy to extend across products)
    product_id = df["product_id"].unique()[0]
    product_df = df[df["product_id"] == product_id].sort_values("date")
    product_df.set_index("date", inplace=True)
    product_df = product_df.asfreq("D")
    series = product_df["daily_sales"]

    adf_results = adf_test(series)

    # 3) EDA plots
    plot_sales_trends(df, product_id, "visualizations/sales_trend.png")

    # 4) Train/Test Split
    train, test = train_test_split_by_time(series, TEST_DAYS)

    # 5) ARIMA Modeling
    order = tune_arima(train)
    arima_model = train_arima(train, order)
    arima_forecast = forecast_arima(arima_model, steps=TEST_DAYS)
    arima_forecast.index = test.index

    arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
    arima_mae = mean_absolute_error(test, arima_forecast)

    # 6) LSTM Modeling
    scaled, scaler = scale_for_lstm(series)
    x_all, y_all = make_lstm_sequences(scaled, LSTM_WINDOW)
    split_index = len(series) - TEST_DAYS - LSTM_WINDOW

    x_train, y_train = x_all[:split_index], y_all[:split_index]
    x_test, y_test = x_all[split_index:], y_all[split_index:]

    lstm_model = build_lstm(LSTM_WINDOW)
    train_lstm(lstm_model, x_train, y_train, epochs=LSTM_EPOCHS, batch=LSTM_BATCH)

    lstm_scaled_preds = forecast_lstm(lstm_model, scaled[:-TEST_DAYS], LSTM_WINDOW, TEST_DAYS)
    lstm_preds = scaler.inverse_transform(lstm_scaled_preds.reshape(-1, 1)).flatten()
    lstm_forecast = pd.Series(lstm_preds, index=test.index)

    lstm_rmse = np.sqrt(mean_squared_error(test, lstm_forecast))
    lstm_mae = mean_absolute_error(test, lstm_forecast)

    # 7) Inventory Optimization
    baseline_rp = compute_reorder_point(train.tail(30), LEAD_TIME_DAYS, safety_z=1.0)
    optimized_rp = compute_reorder_point(lstm_forecast, LEAD_TIME_DAYS, SERVICE_LEVEL_Z)

    baseline_sim = simulate_inventory_policy(test, baseline_rp, initial_inventory=baseline_rp * 1.5)
    optimized_sim = simulate_inventory_policy(test, optimized_rp, initial_inventory=optimized_rp * 1.5)

    baseline_cost = inventory_mismatch_cost(baseline_sim)
    optimized_cost = inventory_mismatch_cost(optimized_sim)
    reduction_pct = (baseline_cost - optimized_cost) / baseline_cost * 100

    # 8) Visualizations
    plot_actual_vs_pred(
        test,
        arima_forecast,
        "visualizations/actual_vs_arima.png",
        "Actual vs ARIMA Forecast",
    )
    plot_actual_vs_pred(
        test,
        lstm_forecast,
        "visualizations/actual_vs_lstm.png",
        "Actual vs LSTM Forecast",
    )
    plot_inventory_levels(
        baseline_sim,
        optimized_sim,
        "visualizations/inventory_comparison.png",
    )

    # 9) Results
    results = {
        "product_id": product_id,
        "adf_test": adf_results,
        "arima_order": order,
        "arima_rmse": arima_rmse,
        "arima_mae": arima_mae,
        "lstm_rmse": lstm_rmse,
        "lstm_mae": lstm_mae,
        "inventory_cost_reduction_pct": reduction_pct,
    }

    with open("data/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
