import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

# Configuration
DYNAMIC_WINDOW_SIZE = 100
ANOMALY_CONTAMINATION = 0.1
FEATURE_COLS = ['speed', 'rpm', 'speed_increase', 'rpm_increase']

def dynamic_speeding_score(row):
    speed_limit = row['dynamic_speed_limit']
    if speed_limit and row['speed'] > speed_limit:
        severity = (row['speed'] - speed_limit) / speed_limit
        return -10 * min(severity, 2.0)  # Cap penalty at 20 points
    return 0

def dynamic_idle_score(row):
    if row['speed'] < 5 and row['rpm'] > row['idle_rpm_threshold']:
        severity = (row['rpm'] - row['idle_rpm_threshold']) / row['idle_rpm_threshold']
        return -5 * min(severity, 2.0)
    return 0

def anomaly_detection_score(features, model, imputer):
    features = imputer.transform(features.reshape(1, -1))
    return -20 if model.predict(features)[0] == -1 else 0

# Load and prepare data
df = pd.read_csv("enhanced_simulated_obd_data.csv")

# Feature engineering with robust NaN handling
df['speed_prev'] = df['speed'].shift(1).fillna(method='bfill')
df['rpm_prev'] = df['rpm'].shift(1).fillna(method='bfill')
df['speed_increase'] = (df['speed'] - df['speed_prev']).clip(-50, 50).fillna(0)
df['rpm_increase'] = (df['rpm'] - df['rpm_prev']).clip(-5000, 5000).fillna(0)
df['speed_drop'] = (df['speed_prev'] - df['speed']).clip(0, 100).fillna(0)

# Dynamic thresholds
df['dynamic_speed_limit'] = df['speed'].rolling(DYNAMIC_WINDOW_SIZE, min_periods=10).quantile(0.85).fillna(80)
df['idle_rpm_threshold'] = df[df['speed'] < 5]['rpm'].rolling(DYNAMIC_WINDOW_SIZE, min_periods=10).quantile(0.90).ffill()

# Prepare features as numpy array
features_array = df[FEATURE_COLS].to_numpy()

# Create and fit imputer
imputer = SimpleImputer(strategy='median')
features_imputed = imputer.fit_transform(features_array)

# Train model
clf = IsolationForest(contamination=ANOMALY_CONTAMINATION, random_state=42)
clf.fit(features_imputed)

# Calculate scores
df['speeding_score'] = df.apply(dynamic_speeding_score, axis=1)
df['idle_score'] = df.apply(dynamic_idle_score, axis=1)
df['harsh_braking_score'] = df['speed_drop'].apply(lambda x: -min(15, x//2.5) if x > 10 else 0)
df['acceleration_score'] = df.apply(lambda x: -8 if x['speed_increase'] > 12 and x['rpm_increase'] > 600 else 0, axis=1)
df['anomaly_score'] = [anomaly_detection_score(row, clf, imputer) 
                      for row in df[FEATURE_COLS].to_numpy()]

# Adjusted scoring weights
score_weights = {
    'speeding_score': 1.0,
    'idle_score': 0.8,
    'harsh_braking_score': 1.2,
    'acceleration_score': 1.1,
    'anomaly_score': 1.5
}

# Calculate weighted scores
for col, weight in score_weights.items():
    df[col] *= weight

# Normalize to 100-point scale
BASE_SCORE = 100
total_penalty = abs(df[list(score_weights.keys())].sum().sum())
max_penalty = len(df) * 50  # Adjust based on your tolerance
final_score = max(0, BASE_SCORE - (total_penalty/max_penalty)*100)

print(f"\nAI-Calculated Driver Score: {final_score:.1f}/100")
print("Behavioral Insights:")
print(f"- Speeding Events: {len(df[df['speeding_score'] < 0])}")
print(f"- Harsh Braking: {len(df[df['harsh_braking_score'] < 0])}")
print(f"- Driving Anomalies: {len(df[df['anomaly_score'] < 0])}")