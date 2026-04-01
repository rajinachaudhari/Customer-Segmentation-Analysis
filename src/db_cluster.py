import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

from pathlib import Path


# -------------------------------
# LOAD SCALED DATA
# -------------------------------
df = pd.read_feather("dataset/processed_data/dbscan_scaled_features.feather")

# reduce memory
df = df.astype("float32")

# -------------------------------
# OUTPUT PATH
# -------------------------------
output_dir = Path("output")
results_dir = output_dir / "dbscan_results"
results_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# SAMPLE DATA (IMPORTANT)
# -------------------------------
sample_size = 100000
df_sample = df.sample(n=sample_size, random_state=42)

print(f"Using sample of size: {sample_size}")

# -------------------------------
# FIND OPTIMAL EPS (AUTO)
# -------------------------------
print("Finding optimal eps using k-distance graph...")

k = 15  # min_samples suggestion

neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(df_sample)

distances, indices = neighbors_fit.kneighbors(df_sample)

# take kth distance
k_distances = np.sort(distances[:, k-1])

# -------------------------------
# AUTO EPS DETECTION (SMART)
# -------------------------------
# Use percentile (robust method)
eps = np.percentile(k_distances, 99.8)

print(f"Auto-selected eps: {eps:.4f}")

# -------------------------------
# PLOT K-DISTANCE GRAPH
# -------------------------------
plt.figure()
plt.plot(k_distances)
plt.axhline(y=eps, linestyle='--')
plt.title("K-Distance Graph")
plt.xlabel("Points sorted")
plt.ylabel("Distance")
plt.savefig(results_dir / "k_distance.png", dpi=150)
plt.show()

# -------------------------------
# RUN DBSCAN (ON SAMPLE ONLY)
# -------------------------------
print("\nRunning DBSCAN...")

dbscan = DBSCAN(
    eps=eps,
    min_samples=k,
    n_jobs=-1
)

labels = dbscan.fit_predict(df_sample)

# -------------------------------
# RESULTS
# -------------------------------
df_sample["cluster"] = labels

print("\nCluster Distribution (DBSCAN):")
print(pd.Series(labels).value_counts())

# count anomalies
anomaly_count = (labels == -1).sum()
print(f"\nAnomalies detected: {anomaly_count}")

# -------------------------------
# LOAD ORIGINAL USER FEATURES
# -------------------------------
user_df = pd.read_csv("dataset/processed_data/user_features.csv")

# align with sample index
user_sample = user_df.loc[df_sample.index].copy()
user_sample["cluster"] = labels


# -------------------------------
# DEFINE DBSCAN FEATURES
# -------------------------------
dbscan_features = [
    "avg_amount",
    "max_amount",
    "amount_std",
    "txn_velocity",
    "median_time_diff",
    "new_payee_rate",
    "unique_receivers",
    "night_txn_ratio",
    "zero_balance_rate",
    "avg_balance_diff",
]

# -------------------------------
# PROFILE (IGNORE -1)
# -------------------------------
cluster_profile = (
    user_sample[user_sample["cluster"] != -1]
    .groupby("cluster")[dbscan_features]
    .mean()
    .fillna(0)
)
print("\nCluster Profile (DBSCAN):")
print(cluster_profile)

# -------------------------------
# PCA VISUALIZATION
# -------------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(df_sample)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=labels,
    palette="tab10",
    s=10
)
plt.title("DBSCAN Clusters (with Noise)")
plt.savefig(results_dir / "dbscan_pca.png")
plt.show()

# -------------------------------
# SAVE RESULTS
# -------------------------------
user_sample.to_csv(
    "dataset/processed_data/dbscan_segmented_sample.csv",
    index=False
)


#-------------------------------
#ANALYSIS FOR DBSCAN
#-------------------------------

#pie chart for noise vs clustered points
noise = (labels == -1).sum()
normal = (labels != -1).sum()

plt.figure()
plt.pie([normal, noise], labels=["Clustered", "Noise"], autopct="%1.1f%%")
plt.title("Noise vs Clustered Points")
plt.savefig(results_dir / "noise_ratio.png")
plt.show()

#histogram of cluster distribution
plt.figure()
sns.histplot(user_df["avg_amount"], bins=50, kde=True)
plt.title("Distribution of Avg Amount")
plt.savefig(results_dir / "avg_amount_dist.png")
plt.show()

#boxplot of each cluster
plt.figure(figsize=(10,6))
sns.boxplot(x="cluster", y="avg_amount", data=user_sample)
plt.title("Avg Amount by Cluster")
plt.savefig(results_dir / "avg_amount_boxplot.png")
plt.show()


#heatmap of cluster profile
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (
    cluster_profile.max() - cluster_profile.min()
)

plt.figure(figsize=(10,5))
sns.heatmap(cluster_profile_norm, cmap="coolwarm")
plt.title("DBSCAN Cluster Heatmap")
plt.savefig(results_dir / "dbscan_heatmap.png")
plt.show()

print("\nDBSCAN pipeline complete!")