# ==================================================
# Histogram của gia tốc dọc (lon_acc)
# ==================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu
df = pd.read_csv("processed/cleaned_data.csv")

print("=" * 50)
print("Acceleration Distribution Analysis")
print("=" * 50)

print(f"Total Records: {len(df):,}")

# ==================================================
# DATA CLEANING
# ==================================================

# Loại bỏ giá trị thiếu
df = df.dropna(subset=['lon_acc', 'lat_acc'])

print(f"Records after cleaning: {len(df):,}")

# ==================================================
# Histogram của gia tốc dọc (lon_acc)
# ==================================================
plt.figure(figsize=(12,6))

plt.hist(
    df['lon_acc'],
    bins=60,
    density=True,
    edgecolor='black'
)

plt.axvline(
    df['lon_acc'].mean(),
    linestyle='--',
    linewidth=2,
    label=f"Mean = {df['lon_acc'].mean():.2f}"
)

plt.title('Longitudinal Acceleration Distribution')
plt.xlabel('Longitudinal Acceleration (m/s²)')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==================================================
# Histogram của gia tốc ngang (lat_acc)
# ==================================================

plt.figure(figsize=(12,6))

plt.hist(
    df['lat_acc'],
    bins=60,
    density=True,
    edgecolor='black'
)

plt.axvline(
    df['lat_acc'].mean(),
    linestyle='--',
    linewidth=2,
    label=f"Mean = {df['lat_acc'].mean():.2f}"
)

plt.title('Lateral Acceleration Distribution')
plt.xlabel('Lateral Acceleration (m/s²)')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==================================================
# ACCELERATION MAGNITUDE
# ==================================================
df['acc_magnitude'] = np.sqrt(
    df['lon_acc']**2 +
    df['lat_acc']**2
)

plt.figure(figsize=(12, 6))

plt.hist(
    df['acc_magnitude'],
    bins=60,
    density=True,
    edgecolor='black'
)

plt.axvline(
    df['acc_magnitude'].mean(),
    linestyle='--',
    linewidth=2,
    label=f"Mean = {df['acc_magnitude'].mean():.2f}"
)

plt.title('Acceleration Magnitude Distribution')
plt.xlabel('Acceleration Magnitude (m/s²)')
plt.ylabel('Density')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ==================================================
# SUMMARY STATISTICS
# ==================================================

print("\n")
print("=" * 50)
print("LONGITUDINAL ACCELERATION STATISTICS")
print("=" * 50)
print(df['lon_acc'].describe())

print("\n")
print("=" * 50)
print("LATERAL ACCELERATION STATISTICS")
print("=" * 50)
print(df['lat_acc'].describe())

print("\n")
print("=" * 50)
print("ACCELERATION MAGNITUDE STATISTICS")
print("=" * 50)
print(df['acc_magnitude'].describe())