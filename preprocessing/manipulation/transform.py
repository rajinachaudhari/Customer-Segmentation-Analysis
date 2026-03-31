
import pandas as pd 
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import re



df = pd.read_csv("dataset/PaySim Synthetic Financial.csv")


# define start date
start_date = pd.to_datetime("2026-01-01 00:00:00")

# convert step to datetime
df["transaction_time"] = start_date + pd.to_timedelta(df["step"] - 1, unit="h")

# print(df.info())
# print(df["transaction_time"].head())
# print(df["transaction_time"].nunique())

df = df.sort_values(["nameOrig", "transaction_time"])  #first ,all row of same sender are grouped together then sorted by earliest transaction time to latest.

#check how many customers have more than 1 transaction
user_txn_counts = df.groupby("nameOrig").size()
customers_more_than_1 = user_txn_counts[user_txn_counts > 1]
print(customers_more_than_1)

df_more_than_1 = df[df["nameOrig"].isin(customers_more_than_1.index)]
print(df_more_than_1)




df_clean = df.copy()

df_clean = df_clean.drop(columns=["step"])
df_clean["amount_log"] = np.log1p(df_clean["amount"])
df_clean["oldbalanceOrg_log"] = np.log1p(df_clean["oldbalanceOrg"])
df_clean["newbalanceOrig_log"] = np.log1p(df_clean["newbalanceOrig"])
df_clean["oldbalanceDest_log"] = np.log1p(df_clean["oldbalanceDest"])
df_clean["newbalanceDest_log"] = np.log1p(df_clean["newbalanceDest"])

def cap_outliers(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    return np.clip(col, Q1 - 1.5*IQR, Q3 + 1.5*IQR)

df_clean["amount_capped"] = cap_outliers(df_clean["amount_log"])
df_clean["oldbalanceOrg_capped"] = cap_outliers(df_clean["oldbalanceOrg_log"])
df_clean["newbalanceOrig_capped"] = cap_outliers(df_clean["newbalanceOrig_log"])
df_clean["oldbalanceDest_capped"] = cap_outliers(df_clean["oldbalanceDest_log"])
df_clean["newbalanceDest_capped"] = cap_outliers(df_clean["newbalanceDest_log"])



#cleaned df analysis
output_dir = Path("output")

processed_dashboard_dir = output_dir / "processed_dashboard"

def create_subplot_grid(total_plots, max_cols=3, figsize_per_plot=(6, 4)):
    cols = min(max_cols, max(1, total_plots))
    rows = int(np.ceil(total_plots / cols))
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows),
    )
    axes = np.array(axes).reshape(-1)
    return fig, axes

numeric_cols = [
    "newbalanceDest_capped",
    "oldbalanceDest_capped",
    "newbalanceOrig_capped",
    "oldbalanceOrg_capped",
    "amount_capped",
]



if numeric_cols:
    # Dashboard 1: Boxplots for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Boxplots for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        sns.boxplot(x=df_clean[col], ax=ax, color="skyblue")
        ax.set_title(col)
        ax.set_xlabel(col)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(processed_dashboard_dir / "boxplot_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Dashboard 2: Histograms for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Histograms for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df_clean[col], kde=False, ax=ax, color="orange", bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(processed_dashboard_dir / "histogram_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Dashboard 3: Density plots for all numeric columns
    fig, axes = create_subplot_grid(len(numeric_cols), max_cols=3, figsize_per_plot=(6, 4))
    fig.suptitle("Density Plots for Numeric Columns", fontsize=18)

    for ax, col in zip(axes, numeric_cols):
        sns.histplot(df_clean[col], kde=True, ax=ax, color="green", bins=30)
        ax.set_title(col)
        ax.set_xlabel(col)

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(processed_dashboard_dir / "density_dashboard.png", dpi=150, bbox_inches="tight")
    plt.show()


output_dir_data= Path("dataset/processed_data")
output_dir_data.mkdir(parents=True, exist_ok=True)

df_clean.to_csv(output_dir_data/ "df_clean.csv", index=False)
