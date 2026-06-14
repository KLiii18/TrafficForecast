import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

df = pd.read_csv("processed/cleaned_data.csv")

features = [
    "speed",
    "avg_speed",
    "traveled_d",
    "lat",
    "lon",
    "lon_acc",
    "lat_acc",
    "time"
]

corr_matrix = df[features].corr()

plt.figure(figsize=(9,7))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
    linewidths=0.5
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.show()

print(corr_matrix["speed"].sort_values(ascending=False))