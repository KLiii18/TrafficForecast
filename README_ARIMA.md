# ARIMA Model Structure

File theo cấu trúc sau:

```text
src/
├── config.py
└── models/
    ├── common/
    │   ├── metrics.py
    │   └── plots.py
    └── arima/
        ├── train.py
        ├── evaluate.py
        ├── forecast.py
        └── utils.py
```

## Cách chạy

Từ thư mục gốc project:

```bash
python -m src.models.arima.train
```

## Input mặc định

```text
src/data/processed/traffic_density_timeseries.csv
```

## Output

Script sẽ tự tạo:

```text
reports/arima/
├── figures/
│   ├── 01_vehicle_count_over_time.png
│   ├── 02_arima_forecast_vs_actual.png
│   ├── 03_arima_residuals.png
│   └── 04_arima_residual_distribution.png
├── tables/
│   ├── adf_test.csv
│   ├── arima_grid_search_results.csv
│   ├── arima_predictions.csv
│   └── arima_final_metrics.csv
├── models/
│   └── best_arima_model.pkl
└── model_summary.txt
```

## Ý nghĩa từng file

- `train.py`: chạy toàn bộ pipeline ARIMA.
- `evaluate.py`: tạo bảng dự báo và tính chỉ số MAE, RMSE, MAPE, R2.
- `forecast.py`: chứa hàm forecast cho test set và forecast tương lai.
- `utils.py`: đọc dữ liệu, chia train/test, ADF test, tạo bộ tham số ARIMA.
- `common/metrics.py`: metric dùng chung cho nhiều model.
- `common/plots.py`: biểu đồ dùng chung cho nhiều model.
