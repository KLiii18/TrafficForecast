"""
Biểu đồ riêng cho Prophet.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def _prepare_output_path(output_path):
    if output_path is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path

def plot_prophet_forecast(test, forecast_df, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(14, 6))

    plt.plot(
        test["ds"],
        test["y"],
        label="Actual",
        linewidth=2
    )

    plt.plot(
        forecast_df["ds"],
        forecast_df["yhat"],
        label="Prophet Forecast",
        linewidth=2
    )

    plt.fill_between(
        forecast_df["ds"],
        forecast_df["yhat_lower"],
        forecast_df["yhat_upper"],
        alpha=0.2,
        label="Forecast Interval"
    )

    plt.title("Prophet Forecast vs Actual")
    plt.xlabel("Time")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_prophet_components(model, forecast, output_path=None):
    """
    Lưu biểu đồ components của Prophet nếu có thể.
    """
    output_path = _prepare_output_path(output_path)

    fig = model.plot_components(forecast)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)


def plot_residuals(result_df, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(12, 4))
    plt.plot(result_df["ds"], result_df["Error"], marker="o")
    plt.axhline(0, linestyle="--")

    plt.title("Prophet Forecast Residuals")
    plt.xlabel("Time")
    plt.ylabel("Actual - Forecast")
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_residual_distribution(result_df, output_path=None):
    output_path = _prepare_output_path(output_path)

    plt.figure(figsize=(8, 4))
    plt.hist(result_df["Error"].dropna(), bins=8, edgecolor="black")

    plt.title("Distribution of Prophet Residuals")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()
