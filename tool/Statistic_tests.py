from scipy.stats import spearmanr, shapiro, pearsonr



def correlation_precision_test(precision_list, full_batch_drift_mean_list):
    stat_drift, p_drift = shapiro(full_batch_drift_mean_list)
    stat_precision, p_precision = shapiro(precision_list)

    print(f"Shapiro-Wilk for full-batch Drift: stat={stat_drift:.4f}, p={p_drift:.4f}")
    print(f"Shapiro-Wilk for Precision: stat={stat_precision:.4f}, p={p_precision:.4f}")

    if p_drift < 0.05 or p_precision < 0.05:
        corr, pval = spearmanr(full_batch_drift_mean_list, precision_list)

        print("\n📊 Spearman Rank Correlation Test")
        print(f"Correlation coefficient (ρ): {corr:.3f}")
        print(f"p-value: {pval:.4f}")
    else:
        corr, pval = pearsonr(full_batch_drift_mean_list, precision_list)

        print("\n📊 Pearson Correlation Test")
        print(f"Correlation coefficient (ρ): {corr:.3f}")
        print(f"p-value: {pval:.4f}")

    if pval < 0.05:
        print("➡️ Significant correlation between drift and precision (p < 0.05)")
        if corr > 0:
            print("📈 Positive correlation – higher drift tends to give higher precision")
        else:
            print("📉 Negative correlation – higher drift tends to give lower precision")
    else:
        print("ℹ️ No significant correlation (p ≥ 0.05)")



def correlation_recall_test(recall_list, full_batch_drift_mean_list):
    stat_drift, p_drift = shapiro(full_batch_drift_mean_list)
    stat_precision, p_precision = shapiro(recall_list)

    print(f"Shapiro-Wilk for full-batch Drift: stat={stat_drift:.4f}, p={p_drift:.4f}")
    print(f"Shapiro-Wilk for Recall: stat={stat_precision:.4f}, p={p_precision:.4f}")

    if p_drift < 0.05 or p_precision < 0.05:
        corr, pval = spearmanr(full_batch_drift_mean_list, recall_list)

        print("\n📊 Spearman Rank Correlation Test")
        print(f"Correlation coefficient (ρ): {corr:.3f}")
        print(f"p-value: {pval:.4f}")
    else:
        corr, pval = pearsonr(full_batch_drift_mean_list, recall_list)

        print("\n📊 Pearson Correlation Test")
        print(f"Correlation coefficient (ρ): {corr:.3f}")
        print(f"p-value: {pval:.4f}")

    if pval < 0.05:
        print("➡️ Significant correlation between drift and recall (p < 0.05)")
        if corr > 0:
            print("📈 Positive correlation – higher drift tends to give higher recall")
        else:
            print("📉 Negative correlation – higher drift tends to give lower recall")
    else:
        print("ℹ️ No significant correlation (p ≥ 0.05)")