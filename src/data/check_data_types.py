import pandas as pd

df = pd.read_csv("processed/cleaned_data.csv")

numeric_cols = [
    "track_id",
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

for col in numeric_cols:

    invalid = pd.to_numeric(
        df[col],
        errors="coerce"
    ).isna().sum()

    results.append({
        "column": col,
        "expected_type": "numeric",
        "invalid_values": invalid
    })

results.append({
    "column": "type",
    "expected_type": "categorical",
    "invalid_values": df["type"].isna().sum()
})

report = pd.DataFrame(results)

print(report)

report.to_csv(
    "processed/data_type_report.csv",
    index=False
)

print(
    "\nSaved: processed/data_type_report.csv"
)