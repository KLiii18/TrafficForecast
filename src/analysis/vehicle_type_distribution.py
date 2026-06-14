import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# Settings
# ==================================================
sns.set_theme(style="whitegrid", context="notebook")

# ==================================================
# Load Dataset
# ==================================================
df = pd.read_csv("processed/cleaned_data.csv")

# ==================================================
# Vehicle-level Dataset
# ==================================================
vehicle_df = (
    df.groupby("track_id")
      .agg({"type": "first"})
      .reset_index()
)

# ==================================================
# Vehicle Type Distribution
# ==================================================
type_count = (
    vehicle_df["type"]
    .value_counts()
    .sort_values(ascending=False)
)

print("=" * 50)
print("Vehicle Type Distribution")
print("=" * 50)
print(type_count)

# Percentage
type_percent = (
    vehicle_df["type"]
    .value_counts(normalize=True) * 100
).round(2)

print("\nPercentage (%)")
print(type_percent)

# ==================================================
# Bar Chart
# ==================================================
plt.figure(figsize=(8,6))

ax = sns.barplot(
    x=type_count.index,
    y=type_count.values
)

# Add value labels
for i, value in enumerate(type_count.values):
    ax.text(
        i,
        value + 1,
        str(value),
        ha="center",
        fontsize=11
    )

plt.title("Vehicle Type Distribution")
plt.xlabel("Vehicle Type")
plt.ylabel("Number of Vehicles")

plt.xticks(rotation=15)

plt.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.show()