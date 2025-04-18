import numpy as np
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.feature_selection import mutual_info_regression


def run_statistic_tests(chunk_drift_mean_list, batch_drift_mean_list, precision_list, recall_list):
    corr, p = spearmanr(chunk_drift_mean_list, recall_list)
    print(f"Spearman: corr={corr:.2f}, p={p:.4f}")

    pre = recall_list[:125]
    post = recall_list[125:]
    stat, p = mannwhitneyu(pre, post, alternative="two-sided")
    print(f"Mann-Whitney: stat={stat:.2f}, p={p:.4f}")

    # Convert lists to numpy arrays for mutual information calculation
    chunk_drift_arr = np.array(chunk_drift_mean_list).reshape(-1, 1)
    batch_drift_arr = np.array(batch_drift_mean_list).reshape(-1, 1)
    precision_arr = np.array(precision_list)
    recall_arr = np.array(recall_list)

    # Mutual information: drift vs precision/recall
    mi_chunk_precision = mutual_info_regression(chunk_drift_arr, precision_arr)[0]
    mi_chunk_recall = mutual_info_regression(chunk_drift_arr, recall_arr)[0]
    mi_batch_precision = mutual_info_regression(batch_drift_arr, precision_arr)[0]
    mi_batch_recall = mutual_info_regression(batch_drift_arr, recall_arr)[0]

    print(f"Mutual Info – Chunk Drift vs Precision: {mi_chunk_precision:.4f}")
    print(f"Mutual Info – Chunk Drift vs Recall:    {mi_chunk_recall:.4f}")
    print(f"Mutual Info – Batch Drift vs Precision: {mi_batch_precision:.4f}")
    print(f"Mutual Info – Batch Drift vs Recall:    {mi_batch_recall:.4f}")