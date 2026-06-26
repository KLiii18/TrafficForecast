"""
Các hàm dự báo bằng ARIMA.
"""

import pandas as pd


def forecast_test_set(model_fit, test_index):
    """
    Dự báo số bước bằng với độ dài tập test.
    """
    forecast = model_fit.forecast(steps=len(test_index))

    forecast = pd.Series(
        forecast.values,
        index=test_index,
        name="Forecast",
    )

    return forecast


def forecast_future(model_fit, steps):
    """
    Dự báo tương lai N bước tiếp theo.
    Lưu ý: hàm này chưa tự tạo time index tương lai.
    """
    forecast = model_fit.forecast(steps=steps)
    forecast.name = "Forecast"

    return forecast
