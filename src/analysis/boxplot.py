import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# Settings
# ==================================================
sns.set_theme(style="whitegrid", context="notebook")

# ==================================================
# Load dataset
# ==================================================
df = pd.read_csv("processed/cleaned_data.csv")

# ==================================================
# Boxplot: Speed & Average Speed
# ==================================================
plt.figure(figsize=(8,6))

box_df = df[["speed", "avg_speed"]]

sns.boxplot(data=box_df)

plt.title("Boxplot of Speed and Average Speed")
plt.ylabel("Speed (km/h)")

plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================
# Detect Outliers using IQR
# ==================================================
print("="*50)
print("Outlier Detection (IQR Method)")
print("="*50)

for col in ["speed", "avg_speed"]:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower) | (df[col] > upper)]

    print(f"\nFeature : {col}")
    print(f"Q1      : {Q1:.2f}")
    print(f"Q3      : {Q3:.2f}")
    print(f"IQR     : {IQR:.2f}")
    print(f"Lower   : {lower:.2f}")
    print(f"Upper   : {upper:.2f}")
    print(f"Outliers: {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")