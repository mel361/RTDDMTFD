import json

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CONSTANT_VALUES import *


with open(PATH_MODEL_CONFIG) as f:
    FRAUD_FEATURES = json.load(f)["fraud_features"]

# Load the data
micro_drift = pd.read_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_MEAN_DRIFTS)
full_drift = pd.read_csv(PATH_FULL_BATCH_TOOL_STATISTICS_MEAN_DRIFTS)
metrics = pd.read_csv(PATH_METRICS, index_col=0)
micro_feature_drift = pd.read_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES)
full_feature_drift = pd.read_csv(PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES)
micro_batch_iterations_times = pd.read_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_ITERATION_TIMES)
full_batch_iterations_times = pd.read_csv(PATH_FULL_BATCH_TOOL_STATISTICS_ITERATION_TIMES)
micro_batch_drift_detection_times = pd.read_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_TIMES)
full_batch_drift_detection_times = pd.read_csv(PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_TIMES)
micro_batch_drift_detection_ids = pd.read_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_IDS)
full_batch_drift_detection_ids = pd.read_csv(PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_IDS)
micro_batch_features_drifing_per_chunk = pd.read_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES_DRIFTING)
full_batch_features_drifing_per_chunk = pd.read_csv(PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES_DRIFTING)

# Extract precision and recall from the metrics DataFrame
precision = metrics["precision"]
recall = metrics["recall"]


# ---- Comparison Drift vs Precision ----
plt.figure(figsize=(8, 5))
sns.regplot(x=micro_drift["Mean drift"], y=precision, label="Micro-batch", lowess=True)
sns.regplot(x=full_drift["Mean drift"], y=precision, label="Full-batch", lowess=True, color="orange")
plt.xlabel("Mean Drift")
plt.ylabel("Precision")
plt.title("Drift vs Precision: Micro vs Full")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "drift_vs_precision_comparison.png"))
plt.show()
plt.close()

# ---- Comparison Drift vs Recall ----
plt.figure(figsize=(8, 5))
sns.regplot(x=micro_drift["Mean drift"], y=recall, label="Micro-batch", lowess=True)
sns.regplot(x=full_drift["Mean drift"], y=recall, label="Full-batch", lowess=True, color="orange")
plt.xlabel("Mean Drift")
plt.ylabel("Recall")
plt.title("Drift vs Recall: Micro vs Full")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "drift_vs_recall_comparison.png"))
plt.show()
plt.close()

# ---- Feature-specific comparison ----
for feature in FRAUD_FEATURES:
    # Drift vs Precision
    plt.figure(figsize=(8, 5))
    sns.regplot(x=micro_feature_drift[feature], y=precision, label="Micro-batch", lowess=True)
    sns.regplot(x=full_feature_drift[feature], y=precision, label="Full-batch", lowess=True, color="orange")
    plt.xlabel(f"{feature} Drift")
    plt.ylabel("Precision")
    plt.title(f"{feature}: Drift vs Precision")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, f"{feature}_drift_vs_precision.png"))
    plt.show()
    plt.close()

    # Drift vs Recall
    plt.figure(figsize=(8, 5))
    sns.regplot(x=micro_feature_drift[feature], y=recall, label="Micro-batch", lowess=True)
    sns.regplot(x=full_feature_drift[feature], y=recall, label="Full-batch", lowess=True, color="orange")
    plt.xlabel(f"{feature} Drift")
    plt.ylabel("Recall")
    plt.title(f"{feature}: Drift vs Recall")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, f"{feature}_drift_vs_recall.png"))
    plt.show()
    plt.close()

# ---- Iteration times comparison ----
plt.figure(figsize=(8, 5))
plt.plot(micro_batch_iterations_times["Iteration times"], label="Micro-batch", marker="o")
plt.plot(full_batch_iterations_times["Iteration times"], label="Full-batch", marker="o", color="orange")
plt.xlabel("Chunk index")
plt.ylabel("Mean Iteration Time (s)")
plt.title("Mean Iteration Time per Chunk: Micro vs Full")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "iteration_times_comparison.png"))
plt.show()
plt.close()

# ---- Drift detection times comparison ----
plt.figure(figsize=(8, 5))
plt.plot(micro_batch_drift_detection_times["Drift detection times"], label="Micro-batch", marker="o")
plt.plot(full_batch_drift_detection_times["Drift detection times"], label="Full-batch", marker="o", color="orange")
plt.xlabel("Test index")
plt.ylabel("Drift Detection Time (s)")
plt.title("Drift Detection Time per Test: Micro-batch vs Full-batch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "drift_detection_times_comparison.png"))
plt.show()
plt.close()

# ---- Drift detection chunk id----
plt.figure(figsize=(8, 5))
plt.plot(micro_batch_drift_detection_ids["Drift detection ids"], label="Micro-batch", marker="o")
plt.plot(full_batch_drift_detection_ids["Drift detection ids"], label="Full-batch", marker="o", color="orange")
plt.xlabel("Test index")
plt.ylabel("Drift Detection Chunk ID")
plt.title("Drift Detection Chunk ID per Test: Micro-batch vs Full-batch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "drift_detection_chunk_id_comparison.png"))
plt.show()
plt.close()

# ---- All features and mean drift vs precision ----
plt.figure(figsize=(10, 6))

# Micro features
for feature in FRAUD_FEATURES:
    sns.regplot(x=micro_feature_drift[feature], y=precision, lowess=True, scatter=False, color="green")

# Full features
for feature in FRAUD_FEATURES:
    sns.regplot(x=full_feature_drift[feature], y=precision, lowess=True, scatter=False, color="red")

# Mean drift
sns.regplot(x=micro_drift["Mean drift"], y=precision, lowess=True, scatter=False, color="blue",
            label="Micro Mean Drift", line_kws={"linewidth": 5.0})
sns.regplot(x=full_drift["Mean drift"], y=precision, lowess=True, scatter=False, color="orange",
            label="Full Mean Drift", line_kws={"linewidth": 5.0})

plt.xlabel("Drift")
plt.ylabel("Precision")
plt.title("Drift vs Precision – Micro (green/blue), Full (red/orange)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "combined_features_drift_vs_precision.png"))
plt.show()
plt.close()

# ---- All features and mean drift vs recall ----
plt.figure(figsize=(10, 6))

# Micro feature
for feature in FRAUD_FEATURES:
    sns.regplot(x=micro_feature_drift[feature], y=recall, lowess=True, scatter=False, color="green")

# Full features
for feature in FRAUD_FEATURES:
    sns.regplot(x=full_feature_drift[feature], y=recall, lowess=True, scatter=False, color="red")

# Mean drift
sns.regplot(x=micro_drift["Mean drift"], y=recall, lowess=True, scatter=False, color="blue",
            label="Micro Mean Drift", line_kws={"linewidth": 5.0})
sns.regplot(x=full_drift["Mean drift"], y=recall, lowess=True, scatter=False, color="orange",
            label="Full Mean Drift", line_kws={"linewidth": 5.0})

plt.xlabel("Drift")
plt.ylabel("Recall")
plt.title("Drift vs Recall – Micro (green/blue), Full (red/orange)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "combined_features_drift_vs_recall.png"))
plt.show()
plt.close()



# Smooth the features drifting and performance values
micro_smoothed = micro_batch_features_drifing_per_chunk["Features drifting"].rolling(window=5, center=True).mean()
full_smoothed = full_batch_features_drifing_per_chunk["Features drifting"].rolling(window=5, center=True).mean()
recall_smoothed = recall.rolling(window=5, center=True).mean()
precision_smoothed = precision.rolling(window=5, center=True).mean()

# ---- Features Drifting vs Recall per Chunk ----
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(micro_smoothed, label="Micro-batch Drifting")
ax1.plot(full_smoothed, label="Full-batch Drifting", color="orange")
ax1.set_xlabel("Chunk index")
ax1.set_ylabel("Features Drifting per Chunk")
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(recall_smoothed, label="Recall", color="green")
ax2.set_ylabel("Recall")
fig.suptitle("Features Drifting vs Recall per Chunk")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "features_drifting_and_recall_per_chunk.png"))
plt.show()
plt.close()


# ---- Features Drifting vs Precision per Chunk ----
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(micro_batch_features_drifing_per_chunk["Features drifting"], label="Micro-batch Drifting")
ax1.plot(full_batch_features_drifing_per_chunk["Features drifting"], label="Full-batch Drifting", color="orange")
ax1.set_xlabel("Chunk index")
ax1.set_ylabel("Features Drifting per Chunk")
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(precision_smoothed, label="Precision", color="green")
ax2.set_ylabel("Precision")
fig.suptitle("Features Drifting vs Precision per Chunk")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON, "features_drifting_and_precision_per_chunk.png"))
plt.show()
plt.close()