# TrafficForecast
### Urban Traffic Pattern Analysis and Congestion Forecasting Using Drone-Based Traffic Monitoring

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg"/>
    <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-green"/>
    <img src="https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange"/>
    <img src="https://img.shields.io/badge/ARIMA-Time%20Series-red"/>
    <img src="https://img.shields.io/badge/Prophet-Forecasting-purple"/>
</p>

---

# Overview

TrafficForecast processes drone-derived vehicle tracking data and produces congestion forecasts using clustering and time-series models. The pipeline focuses on extracting structured traffic features, identifying congestion hotspots, and comparing ARIMA versus Prophet prediction performance.

---

# Data

The traffic dataset is sourced from: https://open-traffic.epfl.ch/

| Column | Description |
|--------|-------------|
| `track_id` | Each vehicle is tracked using a unique identifier. All data points belonging to the same vehicle will have the same `track_id`. |
| `type` | Type of vehicle, for example: Motorcycle, Car, Bus, Heavy Vehicle, ... |
| `traveled_d` | The total distance the vehicle traveled during the tracking period. |
| `avg_speed` | The average speed of the vehicle over the entire travel route. |
| `lat` | The latitude of the vehicle at a specific point in time. |
| `lon` | The longitude of the vehicle at a specific point in time. |
| `speed` | The instantaneous speed of the vehicle at the time the data is recorded. |
| `lon_acc` | Longitudinal acceleration indicates whether the vehicle is accelerating or decelerating in the direction of travel. |
| `lat_acc` | Lateral acceleration reflects movement to the left/right or when the vehicle is cornering. |
| `time` | Time elapsed since vehicle tracking began. |

---

# Feature

The feature set used for modeling includes:

| Feature | Description |
|---------|-------------|
| `timestamp_real` | Real-world timestamp corresponding to the recorded observation. |
| `time_str` | Human-readable representation of the timestamp, mainly used for visualization and reporting. |
| `time_bin_5m` | Timestamp aggregated into 5-minute intervals. This feature serves as the primary time index for time-series forecasting. |
| `unique_track_id` | Globally unique identifier created by combining the timestamp and track ID to avoid duplication across different videos or sessions. |
| `grid_id` | Spatial grid identifier indicating the grid cell where the vehicle is located. It is commonly used for spatial traffic density analysis. |
| `is_crawling` | Binary indicator showing whether the vehicle is moving at crawling speed. A value of 1 indicates slow-moving traffic, while 0 indicates normal movement. |
| `is_hard_braking` | Binary feature indicating whether the vehicle performs hard braking. A value of 1 represents a hard braking event, while 0 means no hard braking. |
| `pce_factor` | Passenger Car Equivalent (PCE) factor used to convert different vehicle types into an equivalent number of passenger cars, enabling standardized traffic flow analysis. For example, a motorcycle may have a factor of 1.0, while a bus may have a larger factor depending on the adopted traffic engineering standard. |

These engineered features support temporal binning, spatial indexing, and traffic behavior classification.

---

# Train / Test

The workflow uses a holdout split of:

- `75%` training
- `25%` testing

This split evaluates model generalization to unseen traffic conditions.

---

# K-means

K-Means is applied to identify traffic congestion hotspots and spatial clusters.

![K-Means hotspot map](src/models/kmeans/kmeans_hotspot_map.png)

**K-Means final metrics:**

- Silhouette Score = `0.3097`
- Davies-Bouldin Index = `1.1522`

---

# Arima

The ARIMA model forecasts traffic density on the processed time-series dataset.

![ARIMA forecast vs actual](reports/arima/figures/02_arima_forecast_vs_actual.png)

**ARIMA final metrics:**

- MAE = `991.7891097332208`
- RMSE = `1340.8440748522846`
- MAPE = `109.78352848281729%`
- R2 = `0.07657161300859172`

---

# Prophet

Prophet is evaluated as a second forecasting approach to compare against ARIMA.

![Prophet forecast vs actual](reports/prophet/figures/01_prophet_forecast_vs_actual.png)

**Prophet final metrics:**

- MAE = `1411.817931776563`
- RMSE = `1531.2018068765954`
- MAPE = `98.23793896304693%`
- R2 = `-0.20423579582625195`

---

# Comparison

The following figure compares observed traffic density with ARIMA and Prophet forecasts.

![ARIMA vs Prophet comparison](reports/arima/figures/05_arima_vs_prophet.png)

---

# Environment Setup

## Setup

```bash
git clone https://github.com/KLiii18/TrafficForecast.git
cd TrafficForecast/src
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## K-means

```bash
python models/kmeans/step1_feature_engineering.py
python models/kmeans/step2_kmeans_advanced.py
```

## Arima

```bash
python -m src.models.arima.train
```

## Prophet

```bash
python -m src.models.prophet.train
```

---

