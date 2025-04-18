import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def print_graphs(chunk_ids, precision_list, recall_list, chunk_drift_mean_list, batch_drift_mean_list, chunk_feature_drifts, batch_feature_drifts):
    plt.figure(figsize=(12, 6))
    plt.plot(chunk_ids, precision_list, label="Precision")
    plt.plot(chunk_ids, recall_list, label="Recall")
    plt.plot(chunk_ids, chunk_drift_mean_list, label="Chunk Drift (mean)")
    plt.plot(chunk_ids, batch_drift_mean_list, label="Batch Drift (mean)")
    plt.xlabel("Chunk ID")
    plt.ylabel("Score")
    plt.title("Precision, Recall & Drift Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for feature in chunk_feature_drifts:
        plt.figure(figsize=(10, 4))
        plt.plot(chunk_ids, chunk_feature_drifts[feature], label=f"{feature} Drift (Chunk)")
        plt.xlabel("Chunk ID")
        plt.ylabel("Drift Score")
        plt.title(f"Chunk Drift Over Time – {feature}")
        plt.ylim(0, 1)
        plt.axhline(0.1, color='red', linestyle='--', label='Drift Threshold (0.1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    for feature in batch_feature_drifts:
        plt.figure(figsize=(10, 4))
        plt.plot(chunk_ids, batch_feature_drifts[feature], label=f"{feature} Drift (Batch)")
        plt.xlabel("Chunk ID")
        plt.ylabel("Drift Score")
        plt.title(f"Batch Drift Over Time – {feature}")
        plt.ylim(0, 1)
        plt.axhline(0.1, color='red', linestyle='--', label='Drift Threshold (0.1)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    for feature in chunk_feature_drifts:
        plt.figure(figsize=(6, 5))
        plt.scatter(chunk_feature_drifts[feature], precision_list, label="Precision", alpha=0.6)
        plt.scatter(chunk_feature_drifts[feature], recall_list, label="Recall", alpha=0.6)
        plt.xlabel(f"Drift Score – {feature} (Chunk)")
        plt.ylabel("Score")
        plt.title(f"{feature} Drift vs Precision/Recall")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    for feature in batch_feature_drifts:
        plt.figure(figsize=(6, 5))
        plt.scatter(batch_feature_drifts[feature], precision_list, label="Precision", alpha=0.6)
        plt.scatter(batch_feature_drifts[feature], recall_list, label="Recall", alpha=0.6)
        plt.xlabel(f"Drift Score – {feature} (batch)")
        plt.ylabel("Score")
        plt.title(f"{feature} Drift vs Precision/Recall")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(chunk_ids, precision_list, label="Precision", linewidth=2)
    plt.plot(chunk_ids, recall_list, label="Recall", linewidth=2)
    plt.plot(chunk_ids, chunk_drift_mean_list, label="Chunk Drift (mean)", linestyle="--")
    plt.plot(chunk_ids, batch_drift_mean_list, label="Batch Drift (mean)", linestyle="--")
    plt.xlabel("Chunk ID")
    plt.ylabel("Score")
    plt.title("Precision, Recall & Drift (Chunk vs Batch) Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(chunk_drift_mean_list, precision_list, alpha=0.7, label="Precision vs Chunk Drift")
    plt.scatter(chunk_drift_mean_list, recall_list, alpha=0.7, label="Recall vs Chunk Drift")
    plt.xlabel("Mean Chunk Drift")
    plt.ylabel("Score")
    plt.title("Precision/Recall vs Chunk Drift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.scatter(batch_drift_mean_list, precision_list, alpha=0.7, label="Precision vs Batch Drift")
    plt.scatter(batch_drift_mean_list, recall_list, alpha=0.7, label="Recall vs Batch Drift")
    plt.xlabel("Mean Batch Drift")
    plt.ylabel("Score")
    plt.title("Precision/Recall vs Batch Drift")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    chunk_drift_df = pd.DataFrame(chunk_feature_drifts)
    batch_drift_df = pd.DataFrame(batch_feature_drifts)

    # Add model performance metrics
    chunk_drift_df["precision"] = precision_list
    chunk_drift_df["recall"] = recall_list
    batch_drift_df["precision"] = precision_list
    batch_drift_df["recall"] = recall_list

    plt.figure(figsize=(14, 6))

    # Chunk drift heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(chunk_drift_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Korrelation: CHUNK-drift vs Precision/Recall")

    # Batch drift heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(batch_drift_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Korrelation: BATCH-drift vs Precision/Recall")

    plt.tight_layout()
    plt.show()