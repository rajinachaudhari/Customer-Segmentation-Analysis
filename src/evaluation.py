from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "output" / "segmented_users.csv"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "segment_profiles.csv"
REPORT_PATH = OUTPUT_DIR / "segment_profiles.txt"


PROFILE_COLUMNS = [
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
	"transaction_intensity",
	"txns_per_active_day",
	"spend_to_balance_ratio",
	"engagement_score",
	"risk_score",
]


def load_segmented_data(path: Path = INPUT_PATH) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(
			f"Segmented dataset not found at {path}. Run src/clustering.py first."
		)
	return pd.read_csv(path)


def summarize_kmeans_segments(df: pd.DataFrame) -> pd.DataFrame:
	"""Build plain-language names for the K-Means segments."""

	available_columns = [column for column in PROFILE_COLUMNS if column in df.columns]
	cluster_summary = df.groupby("kmeans_segment")[available_columns].mean(numeric_only=True)
	cluster_sizes = df["kmeans_segment"].value_counts().sort_index()

	ranking_frame = pd.DataFrame(index=cluster_summary.index)
	ranking_frame["engagement"] = cluster_summary["engagement_score"] if "engagement_score" in cluster_summary else 0
	ranking_frame["activity"] = cluster_summary["active_days_per_month"] if "active_days_per_month" in cluster_summary else 0
	ranking_frame["recency"] = cluster_summary["recency_days"] if "recency_days" in cluster_summary else 0

	power_cluster = ranking_frame[["engagement", "activity"]].sum(axis=1).idxmax() if len(ranking_frame) else None
	dormant_cluster = ranking_frame["recency"].idxmax() if len(ranking_frame) else None

	rows = []
	for cluster_id, row in cluster_summary.iterrows():
		cluster_slice = df[df["kmeans_segment"] == cluster_id]
		if "segment_type" in df.columns:
			dominant_segment_type = cluster_slice["segment_type"].mode(dropna=True)
			dominant_segment_type = dominant_segment_type.iloc[0] if not dominant_segment_type.empty else None
		else:
			dominant_segment_type = None

		if cluster_id == power_cluster:
			name = "Power Users"
			description = "Highly active users deeply embedded in the platform ecosystem."
		elif cluster_id == dormant_cluster:
			name = "Dormant Users"
			description = "Users who were once active but now transact less often and return less frequently."
		else:
			name = "Occasional Users"
			description = "Users who engage with the platform only when they need specific, infrequent actions."

		rows.append(
			{
				"kmeans_segment": cluster_id,
				"segment_name": name,
				"description": description,
				"users": int(cluster_sizes.loc[cluster_id]),
				"dominant_original_segment_type": dominant_segment_type,
				**{f"avg_{column}": row[column] for column in available_columns},
			}
		)

	return pd.DataFrame(rows).sort_values("kmeans_segment")


def summarize_dbscan_segments(df: pd.DataFrame) -> pd.DataFrame:
	"""Profile DBSCAN output, with noise points treated as high-risk anomalies."""

	available_columns = [column for column in PROFILE_COLUMNS if column in df.columns]
	rows = []

	for segment_id, group in df.groupby("dbscan_segment"):
		if "segment_type" in df.columns:
			dominant_segment_type = group["segment_type"].mode(dropna=True)
			dominant_segment_type = dominant_segment_type.iloc[0] if not dominant_segment_type.empty else None
		else:
			dominant_segment_type = None

		if segment_id == -1:
			name = "High-Risk / Anomalous Users"
			description = "Users whose transaction patterns do not match the normal dense behavior groups and should be reviewed separately."
		else:
			name = f"Dense Group {segment_id}"
			description = "A smaller dense behavioral group identified by DBSCAN."

		row = {
			"dbscan_segment": segment_id,
			"segment_name": name,
			"description": description,
			"users": int(len(group)),
			"dominant_original_segment_type": dominant_segment_type,
		}
		for column in available_columns:
			row[f"avg_{column}"] = group[column].mean()
		rows.append(row)

	return pd.DataFrame(rows).sort_values("dbscan_segment")


def build_profile_report(kmeans_profiles: pd.DataFrame, dbscan_profiles: pd.DataFrame) -> str:
	lines = [
		"Segment Definition and Profiling Report",
		"=======================================",
		"",
		"K-Means segments:",
	]

	for _, row in kmeans_profiles.iterrows():
		lines.append(
			f"- {row['segment_name']} (cluster {row['kmeans_segment']}): {row['description']} Users = {row['users']}. Dominant original label = {row['dominant_original_segment_type']}."
		)

	lines.extend(["", "DBSCAN segments:"])

	for _, row in dbscan_profiles.iterrows():
		lines.append(
			f"- {row['segment_name']} (cluster {row['dbscan_segment']}): {row['description']} Users = {row['users']}. Dominant original label = {row['dominant_original_segment_type']}."
		)

	lines.extend(
		[
			"",
			"How to read the results:",
			"- K-Means gives the core business personas.",
			"- DBSCAN highlights the unusual or sparse-pattern users that are most useful for risk review.",
		]
	)

	return "\n".join(lines)


def run_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
	df = load_segmented_data()

	kmeans_profiles = summarize_kmeans_segments(df)
	dbscan_profiles = summarize_dbscan_segments(df)

	kmeans_profiles.to_csv(OUTPUT_DIR / "kmeans_segment_profiles.csv", index=False)
	dbscan_profiles.to_csv(OUTPUT_DIR / "dbscan_segment_profiles.csv", index=False)
	combined_profiles = pd.concat(
		[kmeans_profiles.assign(model="kmeans"), dbscan_profiles.assign(model="dbscan")],
		ignore_index=True,
		sort=False,
	)
	combined_profiles.to_csv(OUTPUT_PATH, index=False)

	report = build_profile_report(kmeans_profiles, dbscan_profiles)
	REPORT_PATH.write_text(report, encoding="utf-8")

	print(report)
	print(f"\nSaved segment profiles to: {OUTPUT_PATH}")
	print(f"Saved profiling report to: {REPORT_PATH}")

	return kmeans_profiles, dbscan_profiles


if __name__ == "__main__":
	run_evaluation()
