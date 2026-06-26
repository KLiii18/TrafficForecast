"""
Các hàm đánh giá dùng chung cho nhiều model:
ARIMA, Prophet, LSTM, XGBoost...
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)


def calculate_metrics(y_true, y_pred) -> dict:
    """
    Tính các chỉ số đánh giá model dự báo time series.

    Parameters
    ----------
    y_true : array-like
        Giá trị thực tế.
    y_pred : array-like
        Giá trị dự báo.

    Returns
    -------
    dict
        MAE, RMSE, MAPE (%), R2.
    """
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE (%)": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "R2": r2_score(y_true, y_pred),
    }
