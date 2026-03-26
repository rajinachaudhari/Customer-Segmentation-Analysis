# import pandas as pd

# pd.set_option('display.max_columns', 500)

# df = pd.read_csv("fintech_wallet_users_cleaned.csv")
# print(df.describe())
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

raw_path = ROOT / "dataset" / "fintech_wallet_users_sample.csv"
clean_path = ROOT / "fintech_wallet_users_cleaned.csv"
output_path = OUTPUT_DIR / "raw_vs_cleaned_dashboard.png"
boxplot_output_path = OUTPUT_DIR / "raw_vs_cleaned_boxplot_dashboard.png"

raw_df = pd.read_csv(raw_path)
clean_df = pd.read_csv(clean_path)

candidate_columns = [
    "monthly_txn_frequency",
    "avg_txn_value_usd",
    "recency_days",
    "active_days_per_month",
    "feature_adoption_score",
    "spend_to_load_ratio",
    "avg_wallet_balance_usd",
    "monthly_topup_frequency",
    "new_payee_rate",
    "failed_txn_rate",
    "txn_time_spread_hours",
    "account_age_days",
]

columns = [
    column
    for column in candidate_columns
    if column in raw_df.columns and column in clean_df.columns
]

if not columns:
    raise ValueError("No common numeric columns were found to compare.")

sns.set_style("whitegrid")

num_columns = 3
num_rows = math.ceil(len(columns) / num_columns)
fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 5 * num_rows))
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for index, column in enumerate(columns):
    ax = axes[index]

    sns.histplot(
        raw_df[column],
        kde=True,
        stat="density",
        bins=30,
        color="#d95f02",
        alpha=0.35,
        label="Raw",
        ax=ax,
    )
    sns.histplot(
        clean_df[column],
        kde=True,
        stat="density",
        bins=30,
        color="#1b9e77",
        alpha=0.35,
        label="Cleaned",
        ax=ax,
    )

    ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")

for index in range(len(columns), len(axes)):
    fig.delaxes(axes[index])

fig.suptitle("Raw vs Cleaned Dataset Comparison", fontsize=18, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved dashboard plot to: {output_path}")

fig, axes = plt.subplots(num_rows, num_columns, figsize=(18, 5 * num_rows))
axes = axes.flatten() if hasattr(axes, "flatten") else [axes]

for index, column in enumerate(columns):
    ax = axes[index]

    box_data = [raw_df[column].dropna(), clean_df[column].dropna()]
    box = ax.boxplot(box_data, vert=False, labels=["Raw", "Cleaned"], patch_artist=True)
    for patch, color in zip(box["boxes"], ["#d95f02", "#1b9e77"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    ax.set_title(column)
    ax.set_xlabel(column)
    ax.set_ylabel("")

for index in range(len(columns), len(axes)):
    fig.delaxes(axes[index])

fig.suptitle("Raw vs Cleaned Dataset Box Plot Comparison", fontsize=18, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(boxplot_output_path, dpi=300, bbox_inches="tight")

print(f"Saved box plot dashboard to: {boxplot_output_path}")
plt.show()
