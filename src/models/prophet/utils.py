"""
Tiện ích riêng cho Prophet.
"""

import pickle
from pathlib import Path

import pandas as pd


def load_prophet_data(data_path, time_col="time_bin_5m", target_col="vehicle_count"):
    """
    Đọc dataset timeseries và chuyển về format chuẩn của Prophet:
        ds: thời gian
        y : giá trị cần dự báo
    """
    df = pd.read_csv(data_path)

    prophet_df = df[[time_col, target_col]].copy()
    prophet_df = prophet_df.rename(columns={time_col: "ds", target_col: "y"})

    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df["y"] = prophet_df["y"].astype(float)

    prophet_df = prophet_df.dropna(subset=["ds", "y"])
    prophet_df = prophet_df.sort_values("ds").reset_index(drop=True)

    return prophet_df


def split_last_day(prophet_df):
    """
    Chia train/test giống ARIMA:
    - Train: toàn bộ các ngày trước ngày cuối.
    - Test : ngày cuối cùng.
    """
    last_day = prophet_df["ds"].dt.normalize().max()

    train = prophet_df[prophet_df["ds"].dt.normalize() < last_day].copy()
    test = prophet_df[prophet_df["ds"].dt.normalize() == last_day].copy()

    return train, test


def split_by_ratio(prophet_df, train_ratio=0.8):
    """
    Chia train/test theo tỷ lệ giống notebook Prophet ban đầu.
    Mặc định train 80%, test 20%.
    """
    split_idx = int(len(prophet_df) * train_ratio)

    train = prophet_df.iloc[:split_idx].copy()
    test = prophet_df.iloc[split_idx:].copy()

    return train, test


def save_pickle(obj, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(obj, f)
