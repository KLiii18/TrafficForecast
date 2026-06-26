"""
Đánh giá kết quả dự báo Prophet.
"""

import pandas as pd

from src.models.common.metrics import calculate_metrics


def build_prediction_result(test, pred):
    """
    Tạo bảng kết quả gồm ds, actual, forecast và error.
    """
    result_df = test.merge(pred, on="ds", how="inner")

    result_df = result_df.rename(columns={
        "y": "Actual",
        "yhat": "Forecast",
        "yhat_lower": "Forecast Lower",
        "yhat_upper": "Forecast Upper",
    })

    result_df["Error"] = result_df["Actual"] - result_df["Forecast"]

    return result_df


def evaluate_forecast(result_df):
    """
    Tính MAE, RMSE, MAPE, R2.
    """
    return calculate_metrics(result_df["Actual"], result_df["Forecast"])
