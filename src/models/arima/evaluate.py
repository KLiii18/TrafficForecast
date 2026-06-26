"""
Đánh giá kết quả dự báo ARIMA.
"""

import pandas as pd

from src.models.common.metrics import calculate_metrics


def build_prediction_result(test, forecast):
    """
    Tạo bảng Actual - Forecast - Error.
    """
    result_df = pd.DataFrame({
        "Actual": test,
        "Forecast": forecast,
    })

    result_df["Error"] = result_df["Actual"] - result_df["Forecast"]

    return result_df


def evaluate_forecast(test, forecast):
    """
    Tính MAE, RMSE, MAPE, R2.
    """
    return calculate_metrics(test, forecast)
