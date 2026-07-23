import pandas as pd
import glob
import os

all_data = []

for file in glob.glob("processed/clean_*.csv"):

    filename = os.path.basename(file)

    parts = filename.replace(".csv", "").split("_")

    print(parts)

    # Tách ngày và giờ từ tên file
    date_part = parts[1]
    start_hour = parts[3]

    start_time = pd.to_datetime(
        date_part + start_hour,
        format="%Y%m%d%H%M"
    )

    df = pd.read_csv(file)

    df["timestamp"] = (
        start_time
        + pd.to_timedelta(df["time"], unit="s")
    )

    all_data.append(df)

# Gộp toàn bộ
full_df = pd.concat(all_data, ignore_index=True)

traffic_density = (
    full_df.groupby(
        pd.Grouper(
            key="timestamp",
            freq="5min"
        )
    )["track_id"]
    .nunique()
    .reset_index(name="traffic_density")
)