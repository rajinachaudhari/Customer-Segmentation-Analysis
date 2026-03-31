import pandas as pd
import numpy as np

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv("dataset/processed_data/df_clean.csv")

# Convert datetime
df["transaction_time"] = pd.to_datetime(df["transaction_time"])

# -------------------------------
# Reduce memory (IMPORTANT)
# -------------------------------
df["type"] = df["type"].astype("category")
df["nameOrig"] = df["nameOrig"].astype("category")
df["nameDest"] = df["nameDest"].astype("category")

# -------------------------------
# Drop unnecessary columns early
# -------------------------------
df = df.drop(columns=[
    "oldbalanceOrg_log", "newbalanceOrig_log",
    "oldbalanceDest_log", "newbalanceDest_log",
    "amount_log",
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest"
])

# -------------------------------
# Sort once (for diff & sequence features)
# -------------------------------
df = df.sort_values(["nameOrig", "transaction_time"])

# -------------------------------
# Time-based columns
# -------------------------------
df["hour"] = df["transaction_time"].dt.hour
df["transaction_date"] = df["transaction_time"].dt.date

# -------------------------------
# Create reusable grouped object
# -------------------------------
grouped = df.groupby("nameOrig")

# -------------------------------
# BASIC FEATURES
# -------------------------------
txn_count = grouped.size().rename("txn_count")

txn_amount = grouped["amount_capped"].agg(
    total_amount="sum",
    avg_amount="mean"
)

latest_time = df["transaction_time"].max()
last_txn = grouped["transaction_time"].max()
recency = (latest_time - last_txn).dt.days.rename("recency_days")

active_days = grouped["transaction_date"].nunique().rename("active_days")

time_spread = grouped["hour"].std().rename("txn_time_spread")

service_div = grouped["type"].nunique().rename("service_diversity")

# -------------------------------
# NEW PAYEE FEATURE
# -------------------------------
df["is_new_payee"] = ~df.duplicated(subset=["nameOrig", "nameDest"])
new_payee_rate = grouped["is_new_payee"].mean().rename("new_payee_rate")

# -------------------------------
# TRANSACTION VELOCITY (TIME DIFF)
# -------------------------------
df["time_diff"] = grouped["transaction_time"].diff().dt.total_seconds()
median_time_diff = grouped["time_diff"].median().rename("median_time_diff")

# -------------------------------
# NIGHT TRANSACTIONS
# -------------------------------
df["is_night"] = df["hour"].isin(range(0, 6)).astype(int)
night_ratio = grouped["is_night"].mean().rename("night_txn_ratio")

# -------------------------------
# TYPE DISTRIBUTION (pivot_table as required)
# -------------------------------
txn_type_dist = pd.pivot_table(
    df,
    index="nameOrig",
    columns="type",
    values="amount_capped",
    aggfunc="count",
    fill_value=0
)

txn_type_ratio = txn_type_dist.div(txn_type_dist.sum(axis=1), axis=0)
txn_type_ratio.columns = [f"type_ratio_{col}" for col in txn_type_ratio.columns]

# -------------------------------
# PEAK HOUR
# -------------------------------
peak_hour = grouped["hour"].agg(
    lambda x: x.mode()[0] if not x.mode().empty else 0
).rename("peak_hour")

# -------------------------------
# AMOUNT BEHAVIOR
# -------------------------------
amount_std = grouped["amount_capped"].std().fillna(0).rename("amount_std")
max_amount = grouped["amount_capped"].max().rename("max_amount")

# -------------------------------
# BALANCE BEHAVIOR
# -------------------------------
df["balance_diff"] = df["oldbalanceOrg_capped"] - df["newbalanceOrig_capped"]
avg_balance_diff = grouped["balance_diff"].mean().rename("avg_balance_diff")

df["zero_balance"] = (df["newbalanceOrig_capped"] == 0).astype(int)
zero_balance_rate = grouped["zero_balance"].mean().rename("zero_balance_rate")

# -------------------------------
# TRANSACTION VELOCITY (HOUR-BASED)
# -------------------------------
time_range = grouped["transaction_time"].agg(["min", "max"])
duration_hours = (time_range["max"] - time_range["min"]).dt.total_seconds() / 3600

txn_velocity = (txn_count / duration_hours).replace([np.inf, -np.inf], 0).fillna(0)
txn_velocity = txn_velocity.rename("txn_velocity")

# -------------------------------
# SOCIAL BEHAVIOR
# -------------------------------
unique_receivers = grouped["nameDest"].nunique().rename("unique_receivers")

# -------------------------------
# FINAL CONCAT (ONLY ONCE)
# -------------------------------
user_df = pd.concat([
    txn_count,
    txn_amount,
    recency,
    active_days,
    time_spread,
    service_div,
    new_payee_rate,
    median_time_diff,
    night_ratio,
    txn_type_ratio,
    peak_hour,
    amount_std,
    max_amount,
    avg_balance_diff,
    zero_balance_rate,
    txn_velocity,
    unique_receivers
], axis=1).reset_index()

# -------------------------------
# FINAL FEATURE
# -------------------------------
user_df["txn_frequency"] = user_df["txn_count"] / user_df["active_days"]
user_df["txn_frequency"] = user_df["txn_frequency"].fillna(0)

# -------------------------------
# FINAL CLEANING
# -------------------------------
user_df = user_df.fillna(0)

# -------------------------------
# SAVE
# -------------------------------
user_df.to_csv("dataset/processed_data/user_features.csv", index=False)

print(user_df.info())

