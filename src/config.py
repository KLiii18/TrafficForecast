"""
Cấu hình đường dẫn chung cho project.

File này giả định bạn chạy lệnh từ thư mục gốc project:
    python src/models/arima/train.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "src" / "data" / "processed" / "traffic_density_timeseries.csv"

# Thư mục output riêng cho ARIMA
REPORT_DIR = PROJECT_ROOT / "reports" / "arima"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"
MODEL_DIR = REPORT_DIR / "models"

# Thư mục output riêng cho Prophet
PROPHET_REPORT_DIR = PROJECT_ROOT / "reports" / "prophet"
PROPHET_FIGURE_DIR = PROPHET_REPORT_DIR / "figures"
PROPHET_TABLE_DIR = PROPHET_REPORT_DIR / "tables"
PROPHET_MODEL_DIR = PROPHET_REPORT_DIR / "models"

TIME_COL = "time_bin_5m"
TARGET_COL = "vehicle_count"
