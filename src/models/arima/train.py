"""
Train baseline ARIMA cho bài toán dự báo vehicle_count.

Cách chạy từ thư mục gốc project:
    python src/models/arima/train.py

Kết quả sẽ được lưu vào:
    reports/arima/
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import (
    DATA_PATH,
    FIGURE_DIR,
    MODEL_DIR,
    REPORT_DIR,
    TABLE_DIR,
    TARGET_COL,
    TIME_COL,
)
from src.models.arima.evaluate import build_prediction_result, evaluate_forecast
from src.models.arima.forecast import forecast_test_set
from src.models.arima.utils import (
    generate_arima_orders,
    load_time_series,
    run_adf_test,
    save_pickle,
    split_last_day,
)
from src.models.common.metrics import calculate_metrics
from src.models.common.plots import (
    plot_forecast,
    plot_residual_distribution,
    plot_residuals,
    plot_time_series,
)


def grid_search_arima(train, test, orders):
    """
    Thử nhiều cấu hình ARIMA và chọn model tốt nhất theo RMSE, sau đó AIC.
    """
    model_results = []

    for order in orders:
        try:
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always")

                model = ARIMA(train, order=order)
                model_fit = model.fit()

                forecast_temp = model_fit.forecast(steps=len(test))
                forecast_temp = pd.Series(
                    forecast_temp.values,
                    index=test.index,
                    name="Forecast",
                )

                metrics = calculate_metrics(test, forecast_temp)

                warning_text = "; ".join(
                    str(warning.message) for warning in caught_warnings
                )

                model_results.append({
                    "order": order,
                    "AIC": model_fit.aic,
                    "BIC": model_fit.bic,
                    **metrics,
                    "Warning": warning_text if warning_text else "No warning",
                })

        except Exception as e:
            model_results.append({
                "order": order,
                "AIC": np.nan,
                "BIC": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "MAPE (%)": np.nan,
                "R2": np.nan,
                "Warning": str(e),
            })

    results_df = pd.DataFrame(model_results)

    results_df = results_df.sort_values(
        by=["RMSE", "AIC"],
        ascending=True,
    ).reset_index(drop=True)

    return results_df


def train_best_arima(train, best_order):
    """
    Train lại ARIMA với bộ tham số tốt nhất.
    """
    model = ARIMA(train, order=best_order)
    model_fit = model.fit()

    return model_fit


def main():
    warnings.filterwarnings("ignore")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    print(f"Data path: {DATA_PATH}")

    ts = load_time_series(
        DATA_PATH,
        time_col=TIME_COL,
        target_col=TARGET_COL,
    )

    print(f"Total samples: {len(ts)}")
    print(f"Time period: {ts.index.min()} -> {ts.index.max()}")

    plot_time_series(
        ts,
        title="Vehicle Count Over Time",
        output_path=FIGURE_DIR / "01_vehicle_count_over_time.png",
    )

    train, test = split_last_day(ts)

    print("\nTrain/Test split")
    print("Train samples:", len(train))
    print("Test samples :", len(test))
    print("Train period :", train.index.min(), "->", train.index.max())
    print("Test period  :", test.index.min(), "->", test.index.max())

    print("\nRunning ADF test...")
    adf_result = run_adf_test(train)
    adf_df = pd.DataFrame([adf_result])
    adf_df.to_csv(TABLE_DIR / "adf_test.csv", index=False)

    print(adf_df.to_string(index=False))

    print("\nGrid search ARIMA...")
    orders = generate_arima_orders(
        p_range=range(0, 4),
        d_range=(0,),
        q_range=range(0, 4),
    )

    results_df = grid_search_arima(train, test, orders)
    results_df.to_csv(TABLE_DIR / "arima_grid_search_results.csv", index=False)

    valid_results = results_df.dropna(subset=["RMSE", "AIC"]).copy()

    if valid_results.empty:
        raise RuntimeError("Không có model ARIMA nào train thành công.")

    best_order = valid_results.iloc[0]["order"]

    print("\nBest ARIMA order:", best_order)
    print("\nTop 5 models:")
    print(valid_results.head().to_string(index=False))

    print("\nTraining best ARIMA model...")
    model_fit = train_best_arima(train, best_order)

    with open(REPORT_DIR / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(str(model_fit.summary()))

    save_pickle(model_fit, MODEL_DIR / "best_arima_model.pkl")

    print("\nForecasting test set...")
    forecast = forecast_test_set(model_fit, test.index)

    prediction_df = build_prediction_result(test, forecast)
    prediction_df.to_csv(TABLE_DIR / "arima_predictions.csv")

    final_metrics = evaluate_forecast(test, forecast)
    metrics_df = pd.DataFrame([{
        "model": "ARIMA",
        "best_order": best_order,
        **final_metrics,
    }])

    metrics_df.to_csv(TABLE_DIR / "arima_final_metrics.csv", index=False)

    print("\nFinal metrics:")
    print(metrics_df.to_string(index=False))

    residuals = prediction_df["Error"]

    plot_forecast(
        train,
        test,
        forecast,
        output_path=FIGURE_DIR / "02_arima_forecast_vs_actual.png",
    )

    plot_residuals(
        residuals,
        output_path=FIGURE_DIR / "03_arima_residuals.png",
    )

    plot_residual_distribution(
        residuals,
        output_path=FIGURE_DIR / "04_arima_residual_distribution.png",
    )

    print("\nDone.")
    print(f"Results saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
