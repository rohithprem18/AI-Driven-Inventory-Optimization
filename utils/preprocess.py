import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller


def handle_missing_and_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["product_id", "date"]).copy()

    # Fill missing sales by interpolation per product.
    df["daily_sales"] = (
        df.groupby("product_id")["daily_sales"]
        .apply(lambda s: s.interpolate().bfill().ffill())
        .reset_index(level=0, drop=True)
    )

    # Cap outliers using IQR per product.
    def cap_outliers_series(s: pd.Series) -> pd.Series:
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return s.clip(lower, upper)

    df["daily_sales"] = df.groupby("product_id")["daily_sales"].transform(cap_outliers_series)
    return df


def adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "used_lag": result[2],
        "n_obs": result[3],
    }


def scale_for_lstm(series: pd.Series) -> tuple[np.ndarray, MinMaxScaler]:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))
    return scaled, scaler


def train_test_split_by_time(series: pd.Series, test_days: int) -> tuple[pd.Series, pd.Series]:
    train = series.iloc[:-test_days]
    test = series.iloc[-test_days:]
    return train, test


def make_lstm_sequences(scaled: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(window, len(scaled)):
        x.append(scaled[i - window : i, 0])
        y.append(scaled[i, 0])
    x_arr = np.array(x).reshape(-1, window, 1)
    y_arr = np.array(y)
    return x_arr, y_arr
