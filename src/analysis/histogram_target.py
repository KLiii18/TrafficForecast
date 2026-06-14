import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processed/cleaned_data.csv")

mean_speed = df["speed"].mean()
median_speed = df["speed"].median()
skew_speed = df["speed"].skew()

print("Mean:", mean_speed)
print("Median:", median_speed)
print("Skewness:", skew_speed)

plt.figure(figsize=(10,6))

sns.histplot(
    df["speed"],
    bins=40,
    kde=True
)

plt.axvline(
    mean_speed,
    linestyle="--",
    label=f"Mean = {mean_speed:.2f}"
)

plt.axvline(
    median_speed,
    linestyle="-",
    label=f"Median = {median_speed:.2f}"
)

plt.title("Histogram of Vehicle Speed")
plt.xlabel("Speed (km/h)")
plt.ylabel("Frequency")

plt.legend()
plt.tight_layout()
plt.show()