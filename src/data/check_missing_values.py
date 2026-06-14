import pandas as pd

df = pd.read_csv("processed/cleaned_data.csv")

print("=" * 60)
print("MISSING VALUES REPORT")
print("=" * 60)

missing = df.isnull().sum()

report = pd.DataFrame({
    "column": missing.index,
    "missing_count": missing.values,
    "missing_percent":
        (missing.values / len(df) * 100).round(4)
})

print(report)

report.to_csv(
    "processed/missing_values_report.csv",
    index=False
)

print("\nSaved: processed/missing_values_report.csv")