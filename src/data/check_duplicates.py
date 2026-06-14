import pandas as pd

df = pd.read_csv("processed/cleaned_data.csv")

print("=" * 60)
print("DUPLICATE REPORT")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Total rows: {len(df):,}")
print(f"Duplicate rows: {duplicates:,}")
print(f"Duplicate percentage: {(duplicates/len(df))*100:.4f}%")