# from pathlib import Path

# import numpy as np
# import pandas as pd
# from sklearn.cluster import DBSCAN, KMeans
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import silhouette_score
# from sklearn.neighbors import NearestNeighbors
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler

# ROOT = Path(__file__).resolve().parents[1]
# INPUT_PATH = ROOT / "fintech_wallet_users_cleaned_engineered.csv"
# OUTPUT_DIR = ROOT / "output"
# OUTPUT_DIR.mkdir(exist_ok=True)
# OUTPUT_PATH = OUTPUT_DIR / "segmented_users.csv"
# REPORT_PATH = OUTPUT_DIR / "clustering_report.txt"


# FEATURE_COLUMNS = [
# 	"monthly_txn_frequency",
# 	"avg_txn_value_usd",
# 	"recency_days",
# 	"active_days_per_month",
# 	"feature_adoption_score",
# 	"spend_to_load_ratio",
# 	"avg_wallet_balance_usd",
# 	"monthly_topup_frequency",
# 	"new_payee_rate",
# 	"failed_txn_rate",
# 	"txn_time_spread_hours",
# 	"total_monthly_spend",
# 	"account_age_days",
# 	"transaction_intensity",
# 	"txns_per_active_day",
# 	"spend_to_balance_ratio",
# 	"high_spender_flag",
# 	"engagement_score",
# 	"risk_score",
# ]


# def load_featured_data(path: Path = INPUT_PATH) -> pd.DataFrame:
# 	"""Load the already-engineered CSV used for clustering."""

# 	return pd.read_csv(path)


# def get_cluster_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
# 	"""Keep only the numeric behavioral columns that exist in the dataset."""

# 	available_columns = [column for column in FEATURE_COLUMNS if column in df.columns]
# 	feature_frame = df[available_columns].copy()
# 	return feature_frame, available_columns


# def build_preprocessor() -> Pipeline:
# 	"""Impute missing values and scale the feature space for clustering."""

# 	return Pipeline(
# 		steps=[
# 			("imputer", SimpleImputer(strategy="median")),
# 			("scaler", StandardScaler()),
# 		]
# 	)


# def select_kmeans_k(X: np.ndarray, k_range: range = range(2, 7)) -> tuple[int, dict[int, float]]:
# 	"""Pick the K-Means cluster count using silhouette score with a bias toward 3 clusters.

# 	The bias toward 3 keeps the final result business-friendly when the score is
# 	close to the best solution, which helps map the result to interpretable
# 	personas such as power, occasional, and dormant users.
# 	"""

# 	scores: dict[int, float] = {}
# 	for k in k_range:
# 		if k >= len(X):
# 			continue
# 		model = KMeans(n_clusters=k, random_state=42, n_init=10)
# 		labels = model.fit_predict(X)
# 		if len(set(labels)) > 1:
# 			scores[k] = silhouette_score(X, labels)

# 	if not scores:
# 		return 3, {3: float("nan")}

# 	best_k = max(scores, key=scores.get)
# 	if 3 in scores and scores[3] >= 0.95 * scores[best_k]:
# 		return 3, scores

# 	return best_k, scores


# def select_dbscan_params(X: np.ndarray) -> tuple[float, int, dict[tuple[float, int], float]]:
# 	"""Choose DBSCAN parameters from the local density structure.

# 	We use a small min_samples value because the goal is to isolate unusual
# 	behavior patterns instead of forcing all users into a dense cluster.
# 	eps is searched across quantiles of the k-distance distribution.
# 	"""

# 	n_samples = len(X)
# 	min_samples = max(5, int(np.log10(max(n_samples, 10)) * 5))
# 	min_samples = min(min_samples, max(5, n_samples - 1))

# 	neighbors = NearestNeighbors(n_neighbors=min_samples)
# 	neighbors.fit(X)
# 	distances, _ = neighbors.kneighbors(X)
# 	kth_distances = np.sort(distances[:, -1])

# 	candidate_quantiles = np.linspace(0.7, 0.95, 6)
# 	eps_candidates = np.unique(np.quantile(kth_distances, candidate_quantiles))

# 	scores: dict[tuple[float, int], float] = {}
# 	for eps in eps_candidates:
# 		model = DBSCAN(eps=float(eps), min_samples=min_samples)
# 		labels = model.fit_predict(X)
# 		non_noise_mask = labels != -1
# 		unique_non_noise = set(labels[non_noise_mask])
# 		if non_noise_mask.sum() < 3 or len(unique_non_noise) < 2:
# 			continue
# 		scores[(float(eps), min_samples)] = silhouette_score(X[non_noise_mask], labels[non_noise_mask])

# 	if not scores:
# 		return float(np.quantile(kth_distances, 0.9)), min_samples, {}

# 	best_params = max(scores, key=scores.get)
# 	return best_params[0], best_params[1], scores


# def fit_kmeans(X: np.ndarray) -> tuple[np.ndarray, KMeans, dict[int, float]]:
# 	"""Fit K-Means using the selected number of clusters."""

# 	best_k, scores = select_kmeans_k(X)
# 	model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
# 	labels = model.fit_predict(X)
# 	return labels, model, scores


# def fit_dbscan(X: np.ndarray) -> tuple[np.ndarray, DBSCAN, dict[tuple[float, int], float]]:
# 	"""Fit DBSCAN using density-based parameter selection."""

# 	eps, min_samples, scores = select_dbscan_params(X)
# 	model = DBSCAN(eps=eps, min_samples=min_samples)
# 	labels = model.fit_predict(X)
# 	return labels, model, scores


# def build_clustering_report(
# 	feature_names: list[str],
# 	kmeans_model: KMeans,
# 	kmeans_scores: dict[int, float],
# 	dbscan_model: DBSCAN,
# 	dbscan_scores: dict[tuple[float, int], float],
# 	kmeans_labels: np.ndarray,
# 	dbscan_labels: np.ndarray,
# ) -> str:
# 	"""Create a plain-language report for the segmentation step."""

# 	kmeans_cluster_count = len(set(kmeans_labels))
# 	dbscan_cluster_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
# 	dbscan_noise_count = int((dbscan_labels == -1).sum())

# 	best_k = kmeans_model.n_clusters
# 	best_k_score = kmeans_scores.get(best_k, float("nan"))
# 	best_dbscan_score = max(dbscan_scores.values()) if dbscan_scores else float("nan")

# 	lines = [
# 		"Segmentation Modeling Report",
# 		"============================",
# 		"",
# 		"Feature selection rationale:",
# 		"- We use behavioral, engagement, financial, and risk columns only.",
# 		"- These columns describe how often users transact, how engaged they are, how much they spend, and how risky or unusual their activity looks.",
# 		"- The raw date column is excluded because clustering works better on numeric behavior signals, not timestamps.",
# 		f"- Features used: {', '.join(feature_names)}",
# 		"",
# 		"K-Means clustering:",
# 		f"- Selected k: {best_k}",
# 		f"- Silhouette score for selected k: {best_k_score:.4f}",
# 		f"- Cluster count: {kmeans_cluster_count}",
# 		"- Why K-Means here: it creates compact, interpretable groups for the general user base.",
# 		"- Parameter choice: k was selected using silhouette score, with a slight bias toward 3 clusters when the score is close to the best result so the segments stay business-friendly.",
# 		"",
# 		"DBSCAN clustering:",
# 		f"- Selected eps: {dbscan_model.eps:.4f}",
# 		f"- Selected min_samples: {dbscan_model.min_samples}",
# 		f"- Best silhouette score across candidates: {best_dbscan_score:.4f}",
# 		f"- Non-noise cluster count: {dbscan_cluster_count}",
# 		f"- Noise points flagged as anomalies: {dbscan_noise_count}",
# 		"- Why DBSCAN here: it detects unusual or high-risk behavior and leaves outliers unassigned instead of forcing them into a normal segment.",
# 		"- Parameter choice: eps was selected from the k-distance distribution, and min_samples was kept small enough to catch sparse anomalous users.",
# 		"",
# 		"Key interpretation:",
# 		"- K-Means tells you what the normal behavioral groups look like.",
# 		"- DBSCAN tells you which users do not fit the normal pattern and should be reviewed separately.",
# 	]
# 	return "\n".join(lines)


# def run_segmentation() -> pd.DataFrame:
# 	"""Run feature engineering, scale the data, fit both clustering models, and save the labeled dataset."""

# 	df = load_featured_data()
# 	feature_frame, feature_names = get_cluster_features(df)

# 	preprocessor = build_preprocessor()
# 	X = preprocessor.fit_transform(feature_frame)

# 	kmeans_labels, kmeans_model, kmeans_scores = fit_kmeans(X)
# 	dbscan_labels, dbscan_model, dbscan_scores = fit_dbscan(X)

# 	segmented_df = df.copy()
# 	segmented_df["kmeans_segment"] = kmeans_labels
# 	segmented_df["dbscan_segment"] = dbscan_labels

# 	segmented_df.to_csv(OUTPUT_PATH, index=False)

# 	report = build_clustering_report(
# 		feature_names=feature_names,
# 		kmeans_model=kmeans_model,
# 		kmeans_scores=kmeans_scores,
# 		dbscan_model=dbscan_model,
# 		dbscan_scores=dbscan_scores,
# 		kmeans_labels=kmeans_labels,
# 		dbscan_labels=dbscan_labels,
# 	)
# 	REPORT_PATH.write_text(report, encoding="utf-8")

# 	print(report)
# 	print(f"\nSaved clustered dataset to: {OUTPUT_PATH}")
# 	print(f"Saved clustering report to: {REPORT_PATH}")

# 	return segmented_df


# if __name__ == "__main__":
# 	run_segmentation()




# self code 

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import seaborn as sns

#load data 
df = pd.read_csv("fintech_wallet_users_cleaned_engineered.csv")


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


#select features for clustering
features = [
    'monthly_txn_frequency',
    'avg_txn_value_usd',
    'total_monthly_spend',
    'active_days_per_month',
    'transaction_intensity',
    'txns_per_active_day',
    'recency_days',
    'account_age_days',
    'feature_adoption_score',
 #   'adoption_level',   categorical feature, not suitable for K-Means without encoding
    'engagement_score',
    'avg_wallet_balance_usd',
    'monthly_topup_frequency',
    'spend_to_load_ratio',
    'spend_to_balance_ratio',
    'txn_time_spread_hours',
    'failed_txn_rate',
    'risk_score',
    'high_spender_flag'
]
#loading the features into X
X = df[features].copy()

#handling skewed features with log transformation to reduce the impact of outliers and make the data more normally distributed for better clustering performance.
skewed_features = [
    'monthly_txn_frequency',
    'total_monthly_spend',
    'avg_wallet_balance_usd',
    'recency_days',
    'avg_txn_value_usd'
]

for col in skewed_features:
    X[col] = np.log1p(X[col])  # log(1 + x)
    
#feature scaling using standardization to ensure that all features contribute equally to the distance calculations in K-Means.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



#determine optimal number of clusters using silhouette score and elbow method
inertia = []

K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig(OUTPUT_DIR / "elbow_method.png", dpi=300, bbox_inches="tight")
plt.show()

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"K={k}, Silhouette Score={score:.3f}")
    
best_k = 2
best_score = -1

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    
    if score > best_score:
        best_score = score
        best_k = k

print(f"Best K based on Silhouette Score: {best_k}")
    
#fit K-Means with the selected number of clusters (e.g., k=4 based on the elbow method and silhouette scores)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(X_scaled)

#cluster labeling
cluster_labels = {
    0: "Core Active Users",
    1: "Casual Users",
    2: "Dormant Users",
    3: "High-Value Risky Users"
}
df['segment_name'] = df['cluster'].map(cluster_labels)

df[['segment_name', 'cluster'] + features].to_csv(
    OUTPUT_DIR / "segmented_users.csv", index=False
)


#checking cluster summary


#cluster summary to understand the average behavior of users in each cluster, which can help in interpreting the segments and tailoring strategies for each group.
cluster_summary = df.groupby('cluster')[features].mean()
print(cluster_summary)

#users in each segment
print(df['segment_name'].value_counts())
#high value users
high_value_users = df[df['segment_name'] == "High-Value Risky Users"]
print(high_value_users.head())
#dormant users
dormant_users = df[df['segment_name'] == "Dormant Users"]
print(dormant_users.head())
	


#visualize clusters using countplot to show the distribution of users across the identified clusters, which helps in understanding the size of each segment and can inform targeted strategies for each group.
sns.countplot(x='cluster', data=df)
plt.title("Cluster Distribution")
plt.savefig(OUTPUT_DIR / "cluster_distribution.png", dpi=300, bbox_inches="tight")
plt.show()

#visualize clusters using PCA to reduce the dimensionality of the feature space to 2D for visualization purposes, allowing us to see how well the clusters are separated and to identify any potential overlaps or outliers in the data.
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis')
plt.title("Clusters Visualization (PCA)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.savefig(OUTPUT_DIR / "clusters_pca.png", dpi=300, bbox_inches="tight")
plt.show()


