import pandas as pd
from glob import glob

# Đọc tất cả file clean
files = glob("processed/cleaned_data.csv")

df_list = []

for file in files:
    print(f"Reading {file}")
    df = pd.read_csv(file)
    df_list.append(df)

# Ghép dữ liệu
df = pd.concat(df_list, ignore_index=True)

# Các biến số cần thống kê
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

# Tính thống kê
summary = pd.DataFrame({
    "Min": df[numeric_cols].min(),
    "Max": df[numeric_cols].max(),
    "Mean": df[numeric_cols].mean(),
    "Median": df[numeric_cols].median(),
    "Std": df[numeric_cols].std()
})

print(summary)

# Lưu ra file
summary.to_csv(
    "processed/descriptive_statistics.csv"
)

print("Saved: processed/descriptive_statistics.csv")