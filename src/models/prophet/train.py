"""
Train baseline Prophet cho bài toán dự báo vehicle_count.

Cách chạy từ thư mục gốc project:
    python -m src.models.prophet.train

Kết quả sẽ được lưu vào:
    reports/prophet/
"""

import warnings

import pandas as pd
from prophet import Prophet

from src.config import (
    DATA_PATH,
    PROPHET_FIGURE_DIR,
    PROPHET_MODEL_DIR,
    PROPHET_REPORT_DIR,
    PROPHET_TABLE_DIR,
    TARGET_COL,
    TIME_COL,
)
from src.models.prophet.evaluate import build_prediction_result, evaluate_forecast
from src.models.prophet.forecast import forecast_test_set
from src.models.prophet.plots import (
    plot_prophet_components,
    plot_prophet_forecast,
    plot_residual_distribution,
    plot_residuals,
)
from src.models.prophet.utils import load_prophet_data, save_pickle, split_last_day


def build_prophet_model():
    """
    Tạo model Prophet baseline.

    Với dataset nhỏ và chỉ có vài ngày dữ liệu, tắt weekly/yearly seasonality
    để tránh model học sai chu kỳ dài hạn không tồn tại trong dữ liệu.
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=False,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
    )

    return model


def train_prophet(train):
    """
    Train Prophet trên tập train.
    """
    model = build_prophet_model()
    model.fit(train)

    return model


def main():
    warnings.filterwarnings("ignore")

    PROPHET_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROPHET_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PROPHET_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PROPHET_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    print(f"Data path: {DATA_PATH}")

    prophet_df = load_prophet_data(
        DATA_PATH,
        time_col=TIME_COL,
        target_col=TARGET_COL,
    )

    print(f"Total samples: {len(prophet_df)}")
    print(f"Time period: {prophet_df['ds'].min()} -> {prophet_df['ds'].max()}")

    train, test = split_last_day(prophet_df)

    print("\nTrain/Test split")
    print("Train samples:", len(train))
    print("Test samples :", len(test))
    print("Train period :", train["ds"].min(), "->", train["ds"].max())
    print("Test period  :", test["ds"].min(), "->", test["ds"].max())

    if train.empty or test.empty:
        raise RuntimeError("Train hoặc test đang rỗng. Hãy kiểm tra lại dữ liệu đầu vào.")

    print("\nTraining Prophet model...")
    model = train_prophet(train)

    save_pickle(model, PROPHET_MODEL_DIR / "best_prophet_model.pkl")

    print("\nForecasting test set...")
    pred = forecast_test_set(model, test)

    result_df = build_prediction_result(test, pred)
    result_df.to_csv(PROPHET_TABLE_DIR / "prophet_predictions.csv", index=False)

    final_metrics = evaluate_forecast(result_df)
    metrics_df = pd.DataFrame([{
        "model": "Prophet",
        **final_metrics,
    }])

    metrics_df.to_csv(PROPHET_TABLE_DIR / "prophet_final_metrics.csv", index=False)

    print("\nFinal metrics:")
    print(metrics_df.to_string(index=False))

    # Lưu forecast đầy đủ để vẽ components
    full_forecast = model.predict(prophet_df[["ds"]])
    full_forecast.to_csv(PROPHET_TABLE_DIR / "prophet_full_forecast.csv", index=False)

    plot_prophet_forecast(
    test=test,
    forecast_df=pred,
    output_path=PROPHET_FIGURE_DIR / "01_prophet_forecast_vs_actual.png",
)

    plot_residuals(
        result_df,
        output_path=PROPHET_FIGURE_DIR / "02_prophet_residuals.png",
    )

    plot_residual_distribution(
        result_df,
        output_path=PROPHET_FIGURE_DIR / "03_prophet_residual_distribution.png",
    )

    try:
        plot_prophet_components(
            model,
            full_forecast,
            output_path=PROPHET_FIGURE_DIR / "04_prophet_components.png",
        )
    except Exception as e:
        print("Cannot save Prophet components plot:", e)

    print("\nDone.")
    print(f"Results saved to: {PROPHET_REPORT_DIR}")


if __name__ == "__main__":
    main()
