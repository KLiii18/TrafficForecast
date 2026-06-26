# Prophet Model Structure

File vào project theo cấu trúc:

```text
src/
├── config.py
└── models/
    ├── common/
    │   ├── metrics.py
    │   └── plots.py
    └── prophet/
        ├── __init__.py
        ├── train.py
        ├── evaluate.py
        ├── forecast.py
        ├── plots.py
        └── utils.py
```

## Cách chạy

Từ thư mục gốc project:

```bash
python -m src.models.prophet.train
```

## Input mặc định

```text
src/data/processed/traffic_density_timeseries.csv
```

## Output

Script sẽ tạo:

```text
reports/prophet/
├── figures/
│   ├── 01_prophet_forecast_vs_actual.png
│   ├── 02_prophet_residuals.png
│   ├── 03_prophet_residual_distribution.png
│   └── 04_prophet_components.png
├── tables/
│   ├── prophet_predictions.csv
│   ├── prophet_final_metrics.csv
│   └── prophet_full_forecast.csv
└── models/
    └── best_prophet_model.pkl
```

## Ghi chú

Phiên bản này chia train/test theo ngày cuối cùng, giống pipeline ARIMA hiện tại. Prophet được cấu hình:

```python
Prophet(
    daily_seasonality=True,
    weekly_seasonality=False,
    yearly_seasonality=False,
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
)
```

Với dataset ít ngày, không nên bật weekly/yearly seasonality vì model không đủ dữ liệu để học các chu kỳ dài.
