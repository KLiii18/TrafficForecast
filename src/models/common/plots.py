"""
Các hàm vẽ biểu đồ dùng chung cho các model.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _prepare_output_path(output_path):
    if output_path is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_time_series(ts: pd.Series, title: str, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(14, 5))
    x = range(len(ts))

    plt.plot(x, ts.values, marker="o", linewidth=1)

    tick_positions = range(0, len(ts), max(1, len(ts) // 8))
    tick_labels = [ts.index[i].strftime("%m-%d %H:%M") for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(ts.name if ts.name else "Value")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_forecast(train, test, forecast, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(14, 5))

    x_train = range(len(train))
    x_test = range(len(train), len(train) + len(test))

    plt.plot(x_train, train.values, label="Train")
    plt.plot(x_test, test.values, label="Actual")
    plt.plot(x_test, forecast.values, label="Forecast")

    plt.axvline(x=len(train) - 1, linestyle="--", label="Train/Test Split")

    plt.title("ARIMA Forecast vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_residuals(residuals, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(12, 4))
    plt.plot(range(len(residuals)), residuals.values, marker="o")
    plt.axhline(0, linestyle="--")

    plt.title("Forecast Residuals on Test Set")
    plt.xlabel("Test Time Step")
    plt.ylabel("Actual - Forecast")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_residual_distribution(residuals, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 4))
    plt.hist(residuals.dropna(), bins=8, edgecolor="black")

    plt.title("Distribution of Forecast Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()
