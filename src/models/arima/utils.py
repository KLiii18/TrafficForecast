"""
Tiện ích riêng cho ARIMA.
"""

import itertools
import pickle
from pathlib import Path

import pandas as pd
from statsmodels.tsa.stattools import adfuller


def load_time_series(data_path, time_col="time_bin_5m", target_col="vehicle_count"):
    """
    Đọc dataset timeseries và trả về một Series có index là thời gian.
    """
    df = pd.read_csv(data_path)

    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values(time_col)
    df = df.set_index(time_col)

    ts = df[target_col].astype(float)
    ts.name = target_col

    return ts


def split_last_day(ts):
    """
    Chia train/test:
    - Train: toàn bộ các ngày trước ngày cuối.
    - Test : ngày cuối cùng.
    """
    last_day = ts.index.normalize().max()

    train = ts[ts.index.normalize() < last_day]
    test = ts[ts.index.normalize() == last_day]

    return train, test


def run_adf_test(series):
    """
    Kiểm định ADF để xem chuỗi có dừng hay không.

    p-value < 0.05 thường được xem là chuỗi dừng.
    """
    result = adfuller(series.dropna())

    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Used Lag": result[2],
        "Number of Observations": result[3],
        "Critical Value 1%": result[4]["1%"],
        "Critical Value 5%": result[4]["5%"],
        "Critical Value 10%": result[4]["10%"],
    }


def generate_arima_orders(p_range=range(0, 4), d_range=(0,), q_range=range(0, 4)):
    """
    Tạo danh sách các bộ tham số ARIMA(p, d, q).
    Notebook ban đầu dùng d = 0.
    """
    return list(itertools.product(p_range, d_range, q_range))


def save_pickle(obj, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(obj, f)
