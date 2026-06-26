"""
Các hàm dự báo bằng Prophet.
"""

import pandas as pd


def forecast_test_set(model, test):
    """
    Dự báo đúng các mốc thời gian trong tập test.

    Cách này tốt hơn make_future_dataframe trong dataset bị thiếu mốc thời gian,
    vì Prophet sẽ dự báo trực tiếp trên ds của test thay vì tự sinh lịch đều 5 phút.
    """
    future = test[["ds"]].copy()
    forecast = model.predict(future)

    pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()

    return pred


def forecast_future(model, periods, freq="5min"):
    """
    Dự báo tương lai N bước sau dữ liệu train.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
