from scipy.stats import spearmanr, shapiro, pearsonr

def correlation_test(list_y, list_x, label_y, label_x):
    stat_x, p_x = shapiro(list_x)
    stat_y, p_y = shapiro(list_y)

    print("\nShapiro-Wilk Normality Test for " + label_x.capitalize() + " and " + label_y.capitalize())
    print("Shapiro-Wilk for " + label_x + f" drift: stat={stat_x:.4f}, p={p_x:.4f}")
    print("Shapiro-Wilk for " + label_y + f": stat={stat_y:.4f}, p={p_y:.4f}")

    if p_x < 0.05 or p_y< 0.05:
        corr, pval = spearmanr(list_x, list_y)

        print("\nSpearman Rank Correlation Test")
        print(f"Correlation coefficient (ρ): {corr:.3f}")
        print(f"p-value: {pval:.4f}")
    else:
        corr, pval = pearsonr(list_x, list_y)

        print("\nPearson Correlation Test")
        print(f"Correlation coefficient (ρ): {corr:.3f}")
        print(f"p-value: {pval:.4f}")

    if pval < 0.05:
        print("Significant correlation between " + label_x + " drift and " + label_y + " (p < 0.05)")
        if corr > 0:
            print("Positive correlation – higher drift tends to give higher " + label_y + "\n\n")
        else:
            print("Negative correlation – higher drift tends to give lower " + label_y + "\n\n")
    else:
        print("No significant correlation (p ≥ 0.05)\n\n")