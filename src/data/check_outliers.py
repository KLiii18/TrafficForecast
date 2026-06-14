import pandas as pd

# Đọc dữ liệu đã xử lý
df = pd.read_csv("processed/cleaned_data.csv")

# Các cột số cần kiểm tra
numeric_cols = [
    "traveled_d",
    "avg_speed",
    "lat",
    "lon",
    "speed",
    "lon_acc",
    "lat_acc",
    "time"
]

results = []

print("=" * 70)
print("OUTLIER DETECTION REPORT (IQR METHOD)")
print("=" * 70)

for col in numeric_cols:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (
        (df[col] < lower_bound)
        | (df[col] > upper_bound)
    ).sum()

    results.append({
        "column": col,
        "outliers": outliers,
        "percentage": round(outliers / len(df) * 100, 4)
    })

report = pd.DataFrame(results)

print(report)

report.to_csv(
    "processed/outlier_report.csv",
    index=False
)

print("\nSaved: processed/outlier_report.csv")

# ==================================================
# GPS VALIDATION
# ==================================================

invalid_lat = (
    (df["lat"] < -90)
    | (df["lat"] > 90)
).sum()

invalid_lon = (
    (df["lon"] < -180)
    | (df["lon"] > 180)
).sum()

print("\nGPS VALIDATION")
print(f"Invalid Latitude : {invalid_lat}")
print(f"Invalid Longitude: {invalid_lon}")