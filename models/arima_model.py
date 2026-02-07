import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def tune_arima(series: pd.Series, p_values=(0, 1, 2), d_values=(0, 1), q_values=(0, 1, 2)) -> tuple:
    best_order = None
    best_aic = np.inf
    for p in p_values:
        for d in d_values:
            for q in q_values:
                try:
                    model = ARIMA(series, order=(p, d, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
    return best_order if best_order is not None else (1, 1, 1)


def train_arima(series: pd.Series, order: tuple) -> ARIMA:
    model = ARIMA(series, order=order)
    return model.fit()


def forecast_arima(model_fit, steps: int) -> pd.Series:
    return model_fit.forecast(steps=steps)
