import pandas as pd
import numpy as np

def feature_engineering(df):
    df = df.copy()

    # Convert date into model-friendly numeric parts if needed
    if "registration_date" in df.columns:
        registration_date = pd.to_datetime(df["registration_date"], errors="coerce")
        df["registration_year"] = registration_date.dt.year.astype("Int64")
        df["registration_month"] = registration_date.dt.month.astype("Int64")
        df["registration_day"] = registration_date.dt.day.astype("Int64")
        df = df.drop(columns=["registration_date"])

    # Directly available / useful columns
    if "monthly_txn_frequency" in df.columns and "avg_txn_value_usd" in df.columns:
        df["transaction_intensity"] = df["monthly_txn_frequency"] * df["avg_txn_value_usd"]

    if "active_days_per_month" in df.columns and "monthly_txn_frequency" in df.columns:
        df["txns_per_active_day"] = df["monthly_txn_frequency"] / df["active_days_per_month"].replace(0, np.nan)

    if "total_monthly_spend" in df.columns and "avg_wallet_balance_usd" in df.columns:
        df["spend_to_balance_ratio"] = df["total_monthly_spend"] / df["avg_wallet_balance_usd"].replace(0, np.nan)

    if "spend_to_load_ratio" in df.columns:
        df["high_spender_flag"] = (df["spend_to_load_ratio"] > 1).astype(int)

    # Engagement score from available engagement columns
    if all(col in df.columns for col in ["active_days_per_month", "feature_adoption_score", "recency_days"]):
        df["engagement_score"] = (
            0.4 * df["active_days_per_month"] +
            0.3 * df["feature_adoption_score"] +
            0.3 * (1 / (df["recency_days"] + 1))
        )

    # Risk score from available risk columns
    if all(col in df.columns for col in ["new_payee_rate", "failed_txn_rate", "txn_time_spread_hours"]):
        df["risk_score"] = (
            0.4 * df["new_payee_rate"] +
            0.4 * df["failed_txn_rate"] +
            0.2 * (df["txn_time_spread_hours"] / 24)
        )

    # Optional: service adoption proxy
    if "feature_adoption_score" in df.columns:
        df["adoption_level"] = pd.cut(
            df["feature_adoption_score"],
            bins=[-np.inf, 1, 3, np.inf],
            labels=["low", "medium", "high"]
        )

    return df

if __name__ == "__main__":
    df = pd.read_csv("fintech_wallet_users_cleaned.csv")
    df_engineered = feature_engineering(df)
    df_engineered.to_csv("fintech_wallet_users_cleaned_engineered.csv", index=False)
    print(df_engineered.head())
    print(df_engineered.columns)