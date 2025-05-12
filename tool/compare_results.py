import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from CONSTANT_VALUES import *

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

# Output-map
output_dir = os.path.join(script_dir, '..', VARIANT_NAME, 'output_graphs', 'comparison')
os.makedirs(output_dir, exist_ok=True)

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
plt.savefig(os.path.join(output_dir, "drift_vs_precision_comparison.png"))
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
plt.savefig(os.path.join(output_dir, "drift_vs_recall_comparison.png"))
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
    plt.savefig(os.path.join(output_dir, f"{feature}_drift_vs_precision.png"))
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
    plt.savefig(os.path.join(output_dir, f"{feature}_drift_vs_recall.png"))
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
plt.savefig(os.path.join(output_dir, "iteration_times_comparison.png"))
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
plt.savefig(os.path.join(output_dir, "drift_detection_times_comparison.png"))
plt.show()
plt.close()


# ---- Features Drifting vs Recall per Chunk ----
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(micro_batch_features_drifing_per_chunk["Features drifting"], label="Micro-batch Drifting", marker="o")
ax1.plot(full_batch_features_drifing_per_chunk["Features drifting"], label="Full-batch Drifting", marker="o", color="orange")
ax1.set_xlabel("Chunk index")
ax1.set_ylabel("Features Drifting per Chunk")
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(recall, label="Recall", marker="x", color="green")
ax2.set_ylabel("Recall")
fig.suptitle("Features Drifting vs Recall per Chunk")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "features_drifting_and_recall_per_chunk.png"))
plt.show()
plt.close()


# ---- Features Drifting vs Precision per Chunk ----
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(micro_batch_features_drifing_per_chunk["Features drifting"], label="Micro-batch Drifting", marker="o")
ax1.plot(full_batch_features_drifing_per_chunk["Features drifting"], label="Full-batch Drifting", marker="o", color="orange")
ax1.set_xlabel("Chunk index")
ax1.set_ylabel("Features Drifting per Chunk")
ax1.grid(True)
ax2 = ax1.twinx()
ax2.plot(precision, label="Precision", marker="x", color="green")
ax2.set_ylabel("Precision")
fig.suptitle("Features Drifting vs Precision per Chunk")
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "features_drifting_and_precision_per_chunk.png"))
plt.show()
plt.close()