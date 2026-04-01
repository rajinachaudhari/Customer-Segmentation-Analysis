import pandas as pd
import numpy as np

from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.stats import zscore

# -------------------------------
# LOAD DATA
# -------------------------------
X_scaled = pd.read_feather("dataset/processed_data/scaled_features.feather")
cluster_df = pd.read_feather("dataset/processed_data/clustered_users.feather")

labels = cluster_df["cluster"]

# -------------------------------
# FIX SAMPLING (IMPORTANT)
# -------------------------------
sample_idx = X_scaled.sample(100000, random_state=42).index

X_sample = X_scaled.loc[sample_idx]
labels_sample = labels.loc[sample_idx]

# -------------------------------
# INTERNAL VALIDATION METRICS
# -------------------------------
print("Evaluating Clusters...\n")

sil_score = silhouette_score(X_sample, labels_sample)
print(f"Silhouette Score: {sil_score:.4f}")

ch_score = calinski_harabasz_score(X_sample, labels_sample)
print(f"Calinski-Harabasz Score: {ch_score:.2f}")

db_score = davies_bouldin_score(X_sample, labels_sample)
print(f"Davies-Bouldin Score: {db_score:.4f}")

# -------------------------------
# CLUSTER SIZE BALANCE
# -------------------------------
print("\nCluster Distribution (%):")
print((labels.value_counts(normalize=True) * 100).round(2))

# -------------------------------
# LOAD USER FEATURES
# -------------------------------
user_df = pd.read_csv("dataset/processed_data/user_features.csv")
user_df["cluster"] = labels

# -------------------------------
# SELECTED FEATURES
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
# FEATURE SEPARATION CHECK
# -------------------------------
print("\nFeature Variance Across Clusters:")
for col in selected_features:
    means = user_df.groupby("cluster")[col].mean()
    print(f"{col}: {means.max() - means.min():.4f}")

# -------------------------------
# CLUSTER PROFILE
# -------------------------------
profile = user_df.groupby("cluster")[selected_features].mean()

# -------------------------------
# Z-SCORE PROFILE (KEY STEP)
# -------------------------------
profile_z = profile.apply(zscore)

print("\nZ-score Profile (IMPORTANT):")
print(profile_z.round(2))

# -------------------------------
# FIND TOP FEATURES PER CLUSTER
# -------------------------------
def get_top_features(profile_z, top_n=3):
    top_features = {}
    for cluster in profile_z.index:
        sorted_features = profile_z.loc[cluster].sort_values(ascending=False)
        top_features[cluster] = sorted_features.head(top_n).index.tolist()
    return top_features

top_features = get_top_features(profile_z)

print("\nTop Features per Cluster:")
for k, v in top_features.items():
    print(f"Cluster {k}: {v}")

# -------------------------------
# SMART CLUSTER LABELING (NO FIXED THRESHOLDS)
# -------------------------------
def label_clusters(profile_z):
    labels_map = {}

    for cluster in profile_z.index:
        row = profile_z.loc[cluster]

        #  Risky Users (strong anomaly signals)
        if (
            row["txn_velocity"] > 1 or
            row["amount_std"] > 1 or
            row["night_txn_ratio"] > 1
        ):
            labels_map[cluster] = "Risky Users"

        #  Dormant Users
        elif row["recency_days"] > 1:
            labels_map[cluster] = "Dormant Users"

        #  Power Users (HIGH ACTIVITY)
        elif (
            row["txn_frequency"] > 1 and
            row["active_days"] > 1
        ):
            labels_map[cluster] = "Power Users"

        #  High Value Users (HIGH SPENDING)
        elif (
            row["total_amount"] > 1 or
            row["avg_amount"] > 1
        ):
            labels_map[cluster] = "High Value Users"

        #  Default
        else:
            labels_map[cluster] = "Occasional Users"

    return labels_map
cluster_labels = label_clusters(profile_z)

print("\nFinal Cluster Labels:")
print(cluster_labels)

# -------------------------------
# ASSIGN SEGMENT TO USERS
# -------------------------------
user_df["segment"] = user_df["cluster"].map(cluster_labels)

print("\nSegment Distribution:")
print(user_df["segment"].value_counts())

# -------------------------------
# USER-LEVEL EXPLANATION (WHY USER IN CLUSTER)
# -------------------------------
centroids = profile  # cluster centers

def explain_user(user, cluster_id):
    centroid = centroids.loc[cluster_id]
    diff = user[selected_features] - centroid
    top_features = diff.abs().sort_values(ascending=False).head(3)
    return top_features

# Example user
sample_user = user_df.iloc[0]
cluster_id = sample_user["cluster"]

print("\n--- USER EXPLANATION ---")
print("User belongs to cluster:", cluster_id)
print("Segment:", cluster_labels[cluster_id])
print("Top reasons:")
print(explain_user(sample_user, cluster_id))

# -------------------------------
# SAVE FINAL OUTPUT
# -------------------------------
user_df.to_csv("dataset/processed_data/final_segmented_users.csv", index=False)

print("\n Evaluation & Profiling Complete!")


# import pandas as pd
# import numpy as np

# from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# from scipy.stats import zscore

# # -------------------------------
# # LOAD DATA
# # -------------------------------
# X_scaled = pd.read_feather("dataset/processed_data/scaled_features.feather")
# cluster_df = pd.read_feather("dataset/processed_data/clustered_users.feather")

# labels = cluster_df["cluster"]

# # -------------------------------
# # FIX SAMPLING (IMPORTANT)
# # -------------------------------
# sample_idx = X_scaled.sample(100000, random_state=42).index

# X_sample = X_scaled.loc[sample_idx]
# labels_sample = labels.loc[sample_idx]

# # -------------------------------
# # INTERNAL VALIDATION METRICS
# # -------------------------------
# print("Evaluating Clusters...\n")

# sil_score = silhouette_score(X_sample, labels_sample)
# print(f"Silhouette Score: {sil_score:.4f}")

# ch_score = calinski_harabasz_score(X_sample, labels_sample)
# print(f"Calinski-Harabasz Score: {ch_score:.2f}")

# db_score = davies_bouldin_score(X_sample, labels_sample)
# print(f"Davies-Bouldin Score: {db_score:.4f}")

# # -------------------------------
# # CLUSTER SIZE BALANCE
# # -------------------------------
# print("\nCluster Distribution (%):")
# print((labels.value_counts(normalize=True) * 100).round(2))

# # -------------------------------
# # LOAD USER FEATURES
# # -------------------------------
# user_df = pd.read_csv("dataset/processed_data/user_features.csv")
# user_df["cluster"] = labels

# # -------------------------------
# # SELECTED FEATURES
# # -------------------------------
# selected_features = [
#     "txn_frequency","avg_amount","total_amount","recency_days",
#     "active_days","txn_time_spread","service_diversity",
#     "new_payee_rate","median_time_diff","night_txn_ratio",
#     "amount_std","max_amount","avg_balance_diff",
#     "zero_balance_rate","txn_velocity"
# ]

# # -------------------------------
# # FEATURE SEPARATION CHECK
# # -------------------------------
# print("\nFeature Variance Across Clusters:")
# for col in selected_features:
#     means = user_df.groupby("cluster")[col].mean()
#     print(f"{col}: {means.max() - means.min():.4f}")

# # -------------------------------
# # CLUSTER PROFILE
# # -------------------------------
# profile = user_df.groupby("cluster")[selected_features].mean()

# # -------------------------------
# # Z-SCORE PROFILE
# # -------------------------------
# profile_z = profile.apply(zscore)

# print("\nZ-score Profile (IMPORTANT):")
# print(profile_z.round(2))

# # -------------------------------
# # TOP FEATURES PER CLUSTER
# # -------------------------------
# def get_top_features(profile_z, top_n=3):
#     top_features = {}
#     for cluster in profile_z.index:
#         sorted_features = profile_z.loc[cluster].sort_values(ascending=False)
#         top_features[cluster] = sorted_features.head(top_n).index.tolist()
#     return top_features

# top_features = get_top_features(profile_z)

# print("\nTop Features per Cluster:")
# for k, v in top_features.items():
#     print(f"Cluster {k}: {v}")

# # -------------------------------
# # SMART UNIQUE LABELING (DATA-DRIVEN)
# # -------------------------------
# def label_clusters_unique(profile_z):

#     labels_map = {}
#     used_labels = set()

#     for cluster in profile_z.index:
#         row = profile_z.loc[cluster]

#         scores = {
#             "Power Users": row["txn_frequency"] + row["active_days"] + row["txn_velocity"],
#             "High Value Users": row["total_amount"] + row["avg_amount"] + row["max_amount"],
#             "Low value Active Users": row["recency_days"],
#             "Risky Users": row["txn_velocity"] + row["amount_std"] + row["night_txn_ratio"],
#             "Occasional Users": -(
#                 abs(row["txn_frequency"]) +
#                 abs(row["total_amount"]) +
#                 abs(row["recency_days"])
#             )
#         }

#         # sort behaviors
#         sorted_behaviors = sorted(scores.items(), key=lambda x: x[1], reverse=True)

#         # assign first unused label
#         for label, _ in sorted_behaviors:
#             if label not in used_labels:
#                 labels_map[cluster] = label
#                 used_labels.add(label)
#                 break

#     return labels_map


# cluster_labels = label_clusters_unique(profile_z)

# print("\nFinal Cluster Labels (UNIQUE & LOGICAL):")
# print(cluster_labels)

# # -------------------------------
# # ASSIGN SEGMENTS
# # -------------------------------
# user_df["segment"] = user_df["cluster"].map(cluster_labels)

# print("\nSegment Distribution:")
# print(user_df["segment"].value_counts())

# # -------------------------------
# # CLUSTER INTERPRETATION PRINT
# # -------------------------------
# print("\n--- CLUSTER INTERPRETATION ---")
# for cluster in profile_z.index:
#     print(f"\nCluster {cluster} → {cluster_labels[cluster]}")
#     print(profile_z.loc[cluster].sort_values(ascending=False).head(5))

# # -------------------------------
# # USER-LEVEL EXPLANATION
# # -------------------------------
# centroids = profile

# def explain_user(user, cluster_id):
#     centroid = centroids.loc[cluster_id]
#     diff = user[selected_features] - centroid
#     return diff.abs().sort_values(ascending=False).head(3)

# sample_user = user_df.iloc[0]
# cluster_id = sample_user["cluster"]

# print("\n--- USER EXPLANATION ---")
# print("User belongs to cluster:", cluster_id)
# print("Segment:", cluster_labels[cluster_id])
# print("Top reasons:")
# print(explain_user(sample_user, cluster_id))

# # -------------------------------
# # SAVE FINAL OUTPUT
# # -------------------------------
# user_df.to_csv("dataset/processed_data/final_segmented_users.csv", index=False)

# print("\n Evaluation & Profiling Complete!")