import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("dataset/processed_data/user_features.csv")

# -------------------------------
# OUTPUT PATH
# -------------------------------
output_dir = Path("dataset")
scaled_data_dir = output_dir / "processed_data"
scaled_data_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# DBSCAN FEATURE SELECTION (RISK-FOCUSED)
# -------------------------------
dbscan_features = [
    "avg_amount",          # high transaction value
    "max_amount",          # extreme spikes (important)
    "amount_std",          # variability in behavior
    
    "txn_velocity",        # rapid transactions
    "median_time_diff",    # time gap between transactions
    
    "new_payee_rate",      # new receiver behavior
    "unique_receivers",    # number of distinct receivers
    
    "night_txn_ratio",     # odd-hour activity
    
    "zero_balance_rate",   # wallet draining behavior
    "avg_balance_diff",    # abnormal balance change
]

# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
X = df[dbscan_features].copy()

# Replace inf values if any
X = X.replace([np.inf, -np.inf], np.nan)

# Fill NaN with 0 (safe for behavioral features)
X = X.fillna(0)

# -------------------------------
# SCALING (VERY IMPORTANT FOR DBSCAN)
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=dbscan_features)

# -------------------------------
# SAVE OUTPUT
# -------------------------------
scaled_features_path = scaled_data_dir / "dbscan_scaled_features.feather"
X_scaled_df.to_feather(scaled_features_path)

# Optional CSV (for debugging)
X_scaled_df.to_csv(scaled_data_dir / "dbscan_scaled_features.csv", index=False)

print("DBSCAN Feature Scaling Complete!")
print(X_scaled_df.head())

# -------------------------------
# QUICK CHECK (OPTIONAL)
# -------------------------------
print("\nFeature Summary:")
print(X.describe())