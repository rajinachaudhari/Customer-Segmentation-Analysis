import pandas as pd
import numpy as np

# Reproducibility
rng = np.random.default_rng(seed=42)

n = 30000
date_pool = pd.date_range("2022-01-01", "2025-12-31", freq="D")

# Define segment distribution
segments = rng.choice(
    ['power', 'casual', 'dormant', 'risk'],
    size=n,
    p=[0.25, 0.40, 0.25, 0.10]
)

df = pd.DataFrame({
    "user_id": range(1, n+1),
    "registration_date": rng.choice(date_pool, size=n),
    "segment_type": segments
})

# Initialize columns
df["monthly_txn_frequency"] = 0
df["avg_txn_value_usd"] = 0.0
df["recency_days"] = 0
df["active_days_per_month"] = 0
df["feature_adoption_score"] = 0
df["spend_to_load_ratio"] = 0.0
df["avg_wallet_balance_usd"] = 0.0
df["monthly_topup_frequency"] = 0
df["new_payee_rate"] = 0.0
df["failed_txn_rate"] = 0.0
df["txn_time_spread_hours"] = 0.0

# Generate data based on segment behavior
for i in range(n):
    seg = df.loc[i, "segment_type"]
    
    if seg == "power":
        df.loc[i, "monthly_txn_frequency"] = rng.integers(40, 80)
        df.loc[i, "avg_txn_value_usd"] = rng.uniform(50, 200)
        df.loc[i, "recency_days"] = rng.integers(1, 10)
        df.loc[i, "active_days_per_month"] = rng.integers(20, 30)
        df.loc[i, "feature_adoption_score"] = rng.integers(4, 6)
        df.loc[i, "spend_to_load_ratio"] = rng.uniform(0.8, 1.2)
        df.loc[i, "avg_wallet_balance_usd"] = rng.uniform(200, 1000)
        df.loc[i, "monthly_topup_frequency"] = rng.integers(10, 25)
        df.loc[i, "new_payee_rate"] = rng.uniform(0.1, 0.3)
        df.loc[i, "failed_txn_rate"] = rng.uniform(0.0, 0.05)
        df.loc[i, "txn_time_spread_hours"] = rng.uniform(10, 24)
        
    elif seg == "casual":
        df.loc[i, "monthly_txn_frequency"] = rng.integers(10, 40)
        df.loc[i, "avg_txn_value_usd"] = rng.uniform(10, 100)
        df.loc[i, "recency_days"] = rng.integers(5, 30)
        df.loc[i, "active_days_per_month"] = rng.integers(10, 20)
        df.loc[i, "feature_adoption_score"] = rng.integers(2, 5)
        df.loc[i, "spend_to_load_ratio"] = rng.uniform(0.5, 1.0)
        df.loc[i, "avg_wallet_balance_usd"] = rng.uniform(50, 300)
        df.loc[i, "monthly_topup_frequency"] = rng.integers(5, 15)
        df.loc[i, "new_payee_rate"] = rng.uniform(0.2, 0.5)
        df.loc[i, "failed_txn_rate"] = rng.uniform(0.01, 0.1)
        df.loc[i, "txn_time_spread_hours"] = rng.uniform(6, 18)
        
    elif seg == "dormant":
        df.loc[i, "monthly_txn_frequency"] = rng.integers(1, 10)
        df.loc[i, "avg_txn_value_usd"] = rng.uniform(5, 50)
        df.loc[i, "recency_days"] = rng.integers(60, 180)
        df.loc[i, "active_days_per_month"] = rng.integers(1, 10)
        df.loc[i, "feature_adoption_score"] = rng.integers(1, 3)
        df.loc[i, "spend_to_load_ratio"] = rng.uniform(0.2, 0.6)
        df.loc[i, "avg_wallet_balance_usd"] = rng.uniform(10, 100)
        df.loc[i, "monthly_topup_frequency"] = rng.integers(1, 5)
        df.loc[i, "new_payee_rate"] = rng.uniform(0.0, 0.3)
        df.loc[i, "failed_txn_rate"] = rng.uniform(0.0, 0.05)
        df.loc[i, "txn_time_spread_hours"] = rng.uniform(2, 12)
        
    elif seg == "risk":
        df.loc[i, "monthly_txn_frequency"] = rng.integers(20, 70)
        df.loc[i, "avg_txn_value_usd"] = rng.uniform(20, 300)
        df.loc[i, "recency_days"] = rng.integers(1, 20)
        df.loc[i, "active_days_per_month"] = rng.integers(5, 25)
        df.loc[i, "feature_adoption_score"] = rng.integers(2, 5)
        df.loc[i, "spend_to_load_ratio"] = rng.uniform(0.5, 1.5)
        df.loc[i, "avg_wallet_balance_usd"] = rng.uniform(50, 800)
        df.loc[i, "monthly_topup_frequency"] = rng.integers(5, 20)
        df.loc[i, "new_payee_rate"] = rng.uniform(0.6, 1.0)  # suspicious
        df.loc[i, "failed_txn_rate"] = rng.uniform(0.2, 0.5)  # high risk
        df.loc[i, "txn_time_spread_hours"] = rng.uniform(0, 24)  # irregular timing

# Add derived feature
df["total_monthly_spend"] = df["monthly_txn_frequency"] * df["avg_txn_value_usd"]
df["account_age_days"] = (
    pd.to_datetime("today") - df["registration_date"]
).dt.days
# Save to CSV
df.to_csv("fintech_wallet_users_sample.csv")

print("Dataset generated successfully!")

print(df["segment_type"].value_counts())