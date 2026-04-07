import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler


df= pd.read_csv('dataset/processed_data/user_features.csv')

output_dir_scaled= Path("dataset")
scaled_data_dir = output_dir_scaled / "processed_data"
scaled_data_dir.mkdir(parents=True, exist_ok=True)

# print(df.info())

# # Find duplicate column names
# duplicates_columns= df.columns[df.columns.duplicated()]

# print("Duplicate column names:", duplicates_columns)



# corr_matrix = df.select_dtypes(include=['int64', 'float64']).corr()

# # Find highly correlated columns(almost identical)
# duplicates = np.where(corr_matrix > 0.8)

# for i, j in zip(*duplicates):
#     if i < j:
#         print(corr_matrix.index[i], "==", corr_matrix.columns[j])

# #result: toatal_amount==avg_amount and, active_days==service_diversity and median_txn_diff.




 #feature scaling 

# feature selection
selected_features = [
    "txn_frequency",
    "avg_amount",
    "total_amount",
    "recency_days",
    "active_days",
    "txn_time_spread",
    "service_diversity",
    "new_payee_rate",
    "median_time_diff",
    "night_txn_ratio",
    "type_ratio_CASH_IN",
    "type_ratio_CASH_OUT",
    "type_ratio_PAYMENT",
    "type_ratio_TRANSFER",
    "amount_std",
    "max_amount",
    "avg_balance_diff",
    "zero_balance_rate",
    "txn_velocity"
]

# Initialize scaler
scaler = StandardScaler()

# Fit and transform
X_scaled = scaler.fit_transform(df[selected_features])

# Convert back to DataFrame (optional, for readability)


X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)
scaled_features_path = scaled_data_dir / "scaled_features.feather"
X_scaled_df.to_feather(scaled_features_path)
print(X_scaled_df.head())
X_scaled_df.to_csv(scaled_data_dir / "scaled_features.csv", index=False)
