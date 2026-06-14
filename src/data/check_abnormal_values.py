import pandas as pd

df = pd.read_csv("processed/cleaned_data.csv")

results = []

# traveled distance
invalid = (df["traveled_d"] < 0).sum()
results.append(["traveled_d < 0", invalid])

# average speed
invalid = (df["avg_speed"] < 0).sum()
results.append(["avg_speed < 0", invalid])

# instantaneous speed
invalid = (df["speed"] < 0).sum()
results.append(["speed < 0", invalid])

# latitude
invalid = (
    (df["lat"] < -90) |
    (df["lat"] > 90)
).sum()
results.append(["invalid latitude", invalid])

# longitude
invalid = (
    (df["lon"] < -180) |
    (df["lon"] > 180)
).sum()
results.append(["invalid longitude", invalid])

# time
invalid = (df["time"] < 0).sum()
results.append(["time < 0", invalid])

# acceleration (ngưỡng vật lý tương đối)
invalid = (
    (df["lon_acc"] < -180) |
    (df["lon_acc"] > 180)
).sum()
results.append(["lon_acc outside [-180,180]", invalid])

invalid = (
    (df["lat_acc"] < -90) |
    (df["lat_acc"] > 90)
).sum()
results.append(["lat_acc outside [-90,90]", invalid])

# vehicle type
valid_types = [
    "Car",
    "Motorcycle",
    "Taxi",
    "Bus",
    "Medium Vehicle",
    "Heavy Vehicle"
]

invalid = (
    df["type"]
    .astype(str)
    .str.fullmatch(r"\d+|\s*")
).sum()

results.append([
    "vehicle type invalid (number only or empty)",
    invalid
])

report = pd.DataFrame(
    results,
    columns=[
        "abnormal_check",
        "count"
    ]
)

report["percentage"] = (
    report["count"] / len(df) * 100
).round(4)

print("=" * 70)
print("ABNORMAL VALUES REPORT")
print("=" * 70)

print(report)

report.to_csv(
    "processed/abnormal_values_report.csv",
    index=False
)

print("\nSaved: processed/abnormal_values_report.csv")