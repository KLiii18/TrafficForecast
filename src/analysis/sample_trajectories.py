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

print("="*50)
print("Vehicle Speed over Time")
print("="*50)

print(f"Total Vehicles : {df['track_id'].nunique()}")
print(f"Total Records  : {len(df):,}")

# ==================================================
# Select Vehicles with Longest Trajectories
# ==================================================
N = 5

trajectory_length = (
    df.groupby("track_id")["time"]
      .max()
      .sort_values(ascending=False)
)

sample_tracks = trajectory_length.head(N).index

print("\nSample Track IDs:")
print(sample_tracks.values)

# ==================================================
# Plot
# ==================================================
plt.figure(figsize=(12,6))

for tid in sample_tracks:

    temp = (
        df[df["track_id"] == tid]
        .sort_values("time")
    )

    plt.plot(
        temp["time"],
        temp["speed"],
        linewidth=2,
        label=f"Track {tid}"
    )

    # Starting point
    plt.scatter(
        temp["time"].iloc[0],
        temp["speed"].iloc[0],
        s=60,
        marker="o"
    )

    # Ending point
    plt.scatter(
        temp["time"].iloc[-1],
        temp["speed"].iloc[-1],
        s=60,
        marker="X"
    )

plt.title("Vehicle Speed over Time (Sample Trajectories)")
plt.xlabel("Time (s)")
plt.ylabel("Speed (km/h)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()