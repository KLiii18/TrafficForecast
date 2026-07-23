"""
Compare forecasting results between ARIMA and Prophet.

Run:
    python -m src.models.common.compare_forecasts
"""

import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    FIGURE_DIR,
    TABLE_DIR,
    PROPHET_TABLE_DIR,
)

# ===============================
# Load prediction results
# ===============================

arima = pd.read_csv(
    TABLE_DIR / "arima_predictions.csv",
    index_col=0,
    parse_dates=True,
)

prophet = pd.read_csv(
    PROPHET_TABLE_DIR / "prophet_predictions.csv",
)

# Prophet lưu ds dưới dạng cột
prophet["ds"] = pd.to_datetime(prophet["ds"])
prophet = prophet.set_index("ds")

# ===============================
# Plot
# ===============================

plt.figure(figsize=(12,6))

plt.plot(
    arima.index,
    arima["Actual"],
    linewidth=2.5,
    label="Observed",
)

plt.plot(
    arima.index,
    arima["Forecast"],
    linewidth=2,
    label="ARIMA",
)

plt.plot(
    prophet.index,
    prophet["Forecast"],
    linestyle="--",
    linewidth=2,
    label="Prophet",
)

plt.title("Comparison of Observed and Predicted Traffic Density Using ARIMA and Prophet")

plt.xlabel("Time")
plt.ylabel("Vehicle Count")

plt.xticks(rotation=45)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "05_arima_vs_prophet.png",
    dpi=300,
)

plt.show()