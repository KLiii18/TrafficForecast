import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("processed/cleaned_data.csv")

sample_df = df.sample(min(50000, len(df)), random_state=42)

plt.figure(figsize=(10,8))

sns.kdeplot(
    data=sample_df,
    x="lon",
    y="lat",
    fill=True,
    cmap="Reds",
    levels=30,
    thresh=0.02
)

plt.title("Traffic Density Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()

plt.show()