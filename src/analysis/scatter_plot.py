# ==================================================
# travle distance vs average speed scatter plot
# ==================================================
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_theme(style="whitegrid", context="notebook")

df = pd.read_csv(
    "processed/cleaned_data.csv"
)

print("=" * 50)
print("Dataset Information")
print("=" * 50)
print(f"Number of timestamps : {len(df):,}")
print(f"Number of vehicles   : {df['track_id'].nunique():,}")
print()

vehicle_df = (
    df.groupby("track_id")
      .agg({
          "type": "first",
          "traveled_d": "first",
          "avg_speed": "first"
      })
      .reset_index()
)

print(vehicle_df.head())
print()

# Vẽ scatter plot
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=vehicle_df,
    x="traveled_d",
    y="avg_speed",
    hue="type",
    alpha=0.7
)

plt.title("Travel Distance vs Average Speed by Vehicle Type")
plt.xlabel("Travel Distance (m)")
plt.ylabel("Average Speed (km/h)")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.show()

# Tính corelation
corr = vehicle_df["traveled_d"].corr(vehicle_df["avg_speed"])
print("=" * 50)
print("Vehicle-level Correlation")
print("=" * 50)
print(f"traveled_d vs avg_speed : {corr:.3f}")
print(corr)

# Regression line
plt.figure(figsize=(10,6))

sns.regplot(
    data=vehicle_df,
    x="traveled_d",
    y="avg_speed",
    scatter_kws={"alpha":0.5},
    line_kws={"color":"red"}
)

plt.title("Travel Distance vs Average Speed")
plt.xlabel("Travel Distance (m)")
plt.ylabel("Average Speed (km/h)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

sample_df = df.sample(min(50000, len(df)), random_state=42)
sample_reg = df.sample(min(10000, len(df)), random_state=42)

# ==================================================
# lon_acc vs speed scatter plot
# ==================================================
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=df,
    x="lon_acc",
    y="speed",
    alpha=0.3
)

plt.title("Longitudinal Acceleration vs Speed")
plt.xlabel("Longitudinal Acceleration (m/s²)")
plt.ylabel("Speed (km/h)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Correlation
corr = df["lon_acc"].corr(df["speed"])
print(f"lon_acc vs speed       : {corr:.3f}")

# Regression line
plt.figure(figsize=(10,6))

sns.regplot(
    data=sample_reg,
    x="lon_acc",
    y="speed",
    scatter_kws={"alpha":0.2},
    line_kws={"color": "red"}
)

plt.title("Regression: Longitudinal Acceleration vs Speed")
plt.xlabel("Longitudinal Acceleration (m/s²)")
plt.ylabel("Speed (km/h)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ==================================================
# lat_acc vs speed scatter plot
# ==================================================
plt.figure(figsize=(10,6))

sns.scatterplot(
    data=sample_df,
    x="lat_acc",
    y="speed",
    alpha=0.3
)

plt.title("Lateral Acceleration vs Speed")
plt.xlabel("Lateral Acceleration (m/s²)")
plt.ylabel("Speed (km/h)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Correlation
corr = df["lat_acc"].corr(df["speed"])
print(f"lat_acc vs speed       : {corr:.3f}")

# Regression line
plt.figure(figsize=(10,6))

sns.regplot(
    data=sample_reg,
    x="lat_acc",
    y="speed",
    scatter_kws={"alpha":0.2},
    line_kws={"color": "red"}
)

plt.title("Regression: Lateral Acceleration vs Speed")
plt.xlabel("Lateral Acceleration (m/s²)")
plt.ylabel("Speed (km/h)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

sample_tracks = df["track_id"].drop_duplicates().sample(5, random_state=42)

plt.figure(figsize=(12,6))

for tid in sample_tracks:
    temp = df[df["track_id"] == tid]
    plt.plot(temp["time"], temp["speed"], label=f"Track {tid}")

plt.title("Vehicle Speed over Time")
plt.xlabel("Time (s)")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print()
print("=" * 50)
print("Timestamp-level Correlation")
print("=" * 50)

features = [
    "lon_acc",
    "lat_acc",
    "time",
    "lat",
    "lon"
]

for feature in features:
    corr = df[feature].corr(df["speed"])
    print(f"{feature:15s}: {corr:.3f}")