import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from pathlib import Path


# -------------------------------
# LOAD SCALED DATA
# -------------------------------
df = pd.read_feather("dataset/processed_data/scaled_features.feather")

# reduce memory
df = df.astype("float32")

# -------------------------------
# OUTPUT PATH
# -------------------------------
output_dir = Path("output")
results_dir = output_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

# -------------------------------
# SAMPLE FOR FAST EVALUATION
# -------------------------------
sample = df.sample(n=100000, random_state=42)

# -------------------------------
# FIND BEST K
# -------------------------------
inertia = []
sil_scores = []

K_range = range(2, 15)

print("Finding optimal K...")

for k in K_range:
    model = MiniBatchKMeans(
        n_clusters=k,
        batch_size=10000,
        random_state=42
    )

    model.fit(sample)

    inertia.append(model.inertia_)

    labels = model.predict(sample)
    sil = silhouette_score(sample, labels)
    sil_scores.append(sil)

    print(f"K={k}, Silhouette={sil:.4f}")

# -------------------------------
# PLOT ELBOW
# -------------------------------
plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.xlabel("K")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.savefig(results_dir / "elbow.png", dpi=150)
plt.show()

# -------------------------------
# PLOT SILHOUETTE
# -------------------------------
plt.figure()
plt.plot(K_range, sil_scores, marker='o')
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score vs K")
plt.savefig(results_dir / "silhouette.png", dpi=150)
plt.show()

# -------------------------------
# SELECT BEST K
# -------------------------------
best_k = K_range[np.argmax(sil_scores)]
print(f"\nBest K selected: {best_k}")

# -------------------------------
# TRAIN FINAL MODEL
# -------------------------------
print("Training final model on full dataset...")

final_model = MiniBatchKMeans(
    n_clusters=best_k,
    batch_size=10000,
    random_state=42
)

final_model.fit(df)

print("Training complete!")

# -------------------------------
# ASSIGN CLUSTERS
# -------------------------------
df["cluster"] = final_model.labels_

print("\nCluster Distribution:")
print(df["cluster"].value_counts())

# -------------------------------
# SAVE CLUSTERED DATA
# -------------------------------
df.to_feather("dataset/processed_data/clustered_users.feather")

# -------------------------------
# EVALUATE FINAL MODEL
# -------------------------------
sample_labels = final_model.predict(sample)
final_score = silhouette_score(sample, sample_labels)

print(f"\nFinal Silhouette Score: {final_score:.4f}")


# =========================================================
# ========= FEATURE PROFILING + VISUALIZATION =============
# =========================================================

# -------------------------------
# LOAD ORIGINAL USER FEATURES
# -------------------------------
user_df = pd.read_csv("dataset/processed_data/user_features.csv")

# load cluster labels
cluster_df = pd.read_feather("dataset/processed_data/clustered_users.feather")

# FIX INDEX ALIGNMENT
user_df = user_df.reset_index(drop=True)
cluster_df = cluster_df.reset_index(drop=True)

user_df["cluster"] = cluster_df["cluster"]

# -------------------------------
# DEFINE FEATURES
# -------------------------------
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

# -------------------------------
# CLUSTER PROFILE
# -------------------------------
cluster_profile = user_df.groupby("cluster")[selected_features].mean()

print("\nCluster Profile:")
print(cluster_profile)

# -------------------------------
# NORMALIZED PROFILE
# -------------------------------
cluster_profile_norm = (cluster_profile - cluster_profile.min()) / (
    cluster_profile.max() - cluster_profile.min()
)

print("\nNormalized Profile:")
print(cluster_profile_norm)

# -------------------------------
# SEGMENT NAMING (EDIT LATER)
# -------------------------------
cluster_map = {
    0: "Power Users",
    1: "Low Activity Users",
    2: "High Value Users",
    3: "Dormant Users",
    4: "Risky Users"
}

user_df["segment"] = user_df["cluster"].map(cluster_map)

print("\nSegment Distribution:")
print(user_df["segment"].value_counts())

# -------------------------------
# PCA VISUALIZATION
# -------------------------------
X_scaled = pd.read_feather("dataset/processed_data/scaled_features.feather")

# USE SAME SAMPLE
sample_idx = sample.index

X_sample = X_scaled.loc[sample_idx]
cluster_sample = cluster_df.loc[sample_idx, "cluster"]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sample)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=cluster_sample,
    palette="tab10",
    s=10
)
plt.title("Cluster Visualization (PCA)")
plt.savefig(results_dir / "pca_clusters.png")
plt.show()

# -------------------------------
# HEATMAP
# -------------------------------
plt.figure(figsize=(12,6))
sns.heatmap(cluster_profile_norm, cmap="coolwarm", annot=False)
plt.title("Cluster Feature Heatmap")
plt.savefig(results_dir / "cluster_heatmap.png")
plt.show()

# -------------------------------
# CLUSTER SIZE
# -------------------------------
plt.figure()
user_df["cluster"].value_counts().plot(kind="bar")
plt.title("Cluster Size Distribution")
plt.savefig(results_dir / "cluster_size.png")
plt.show()

# -------------------------------
# SAVE FINAL OUTPUT
# -------------------------------
user_df.to_csv("dataset/processed_data/final_segmented_users.csv", index=False)

print("\nPipeline complete!")