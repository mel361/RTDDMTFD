import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score, precision_score

# Function to print feature information
def print_feature_info():
    missing = fraud_data.isnull().sum()
    print("\nMissing values per column:\n", missing)
    print("\nMissing value percentage per column:\n", (fraud_data.isnull().mean() * 100).round(2))

    duplicates = fraud_data.duplicated().sum()
    print(f"Number of duplicated rows: {duplicates}")

    print("\nData types:\n", X.dtypes)



# Load the data
fraud_data = pd.read_csv('../data/FiFAR/Base.csv')
fraud_features = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w']


# Split the data into features and target variable
y = fraud_data['fraud_bool']
X = fraud_data[fraud_features]

for i in range(1, 9):
    # Split the data into training and testing sets
    train_size = int(len(X) * (i/10))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    # Resample the training data using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(train_X, train_y)

    # Train a Random Forest Classifier
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_resampled, y_resampled)

    # Print the target counts
    print(train_y.value_counts())

    probabilities = model.predict_proba(test_X.head(40000))[:, 1]
    thresholds = np.arange(0.000, 1.000, 0.001)

    print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    print("-" * 40)

    best_percision = (0, 0, 0, 0)
    best_f1Score = (0, 0, 0, 0)
    best_recall = (0, 0, 0, 0)
    for threshold in thresholds:
        preds = (probabilities > threshold).astype(int)
        precision = precision_score(test_y.head(40000), preds, zero_division=0)
        recall = recall_score(test_y.head(40000), preds, zero_division=0)
        f1 = f1_score(test_y.head(40000), preds, zero_division=0)
        if f1 > best_f1Score[3]: best_f1Score = (threshold, precision, recall, f1)
        if precision > best_percision[1]: best_percision = (threshold, precision, recall, f1)
        if recall > best_recall[2]: best_recall = (threshold, precision, recall, f1)
        print(f"{threshold:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

    # Print the best thresholds
    print("\nBest F1 Score: ", best_f1Score[0], "; With Precision: ", best_f1Score[1], ";  Recall: ", best_f1Score[2], ";  F1: ", best_f1Score[3])
    print("Best Precision: ", best_percision[0], "; With Precision: ", best_percision[1], ";  Recall: ", best_percision[2], ";  F1: ", best_percision[3])
    print("Best Recall: ", best_recall[0], "; With Precision: ", best_recall[1], ";  Recall: ", best_recall[2], ";  F1: ", best_recall[3])

    best_threshold = best_f1Score[0]

    # Column mapping for Evidently
    column_mapping = ColumnMapping(
        numerical_features=fraud_features,
        categorical_features=[],
    )
    # Create a report with DataDriftPreset
    report = Report([
        DataDriftPreset(),
    ])


    # Collect metrics for the graphs
    chunk_ids = []
    precision_list = []
    recall_list = []
    chunk_drift_mean_list = []
    batch_drift_mean_list = []

    # To keep track of the drift scores of each feature
    chunk_feature_drifts = {}  # Dictionary to store chunk feature drifts
    batch_feature_drifts = {}

    # Set the chunk size
    chunk_size = 4000
    n = 1
    # Simulate real-time data drift monitoring
    for i in range(0, len(test_X), chunk_size):
        print("Processing chunk: ", n, "////////////////////////////////////////")
        current_chunk = test_X.iloc[i:i + chunk_size]
        current_chunk_target = test_y.iloc[i:i + chunk_size]
        current_batch = test_X.iloc[0:i + chunk_size]

        probabilities = model.predict_proba(current_chunk)[:, 1]
        predictions = (probabilities > best_threshold).astype(int)

        chunk_ids.append(n)

        # Calculate metrics
        precision = precision_score(current_chunk_target, predictions, zero_division=0)
        recall = recall_score(current_chunk_target, predictions, zero_division=0)
        precision_list.append(precision)
        recall_list.append(recall)

        # Empty lists to store drift scores
        chunk_drift_scores = []
        batch_drift_scores = []

        print("Precision: ", precision, "Recall: ", recall)

        report.run(
            current_data=current_chunk,
            reference_data=train_X,
            column_mapping=column_mapping
        )

        resultChunk = report.as_dict()
        report.run(
            current_data=current_batch,
            reference_data=train_X,
            column_mapping=column_mapping
        )
        resultBatch = report.as_dict()

        data_drift = False

        # Check for data drift in the chunk
        print("Checking for drift in chunk")
        for metric in resultChunk["metrics"]:
            if metric["metric"] == "DataDriftTable":
                drift_by_columns = metric["result"].get("drift_by_columns", {})
                for feature_name, feature_data in drift_by_columns.items():
                    drift_score = feature_data["drift_score"]
                    chunk_drift_scores.append(drift_score)

                    # Store the drift score for the feature
                    if drift_score > 0.1:
                        print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                        data_drift = True

                    if feature_name not in chunk_feature_drifts:
                        chunk_feature_drifts[feature_name] = []
                    chunk_feature_drifts[feature_name].append(drift_score)


        # Check for data drift in the batch
        print("Checking for drift in batch")
        for metric in resultBatch["metrics"]:
            if metric["metric"] == "DataDriftTable":
                drift_by_columns = metric["result"].get("drift_by_columns", {})
                for feature_name, feature_data in drift_by_columns.items():
                    drift_score = feature_data["drift_score"]
                    batch_drift_scores.append(drift_score)
                    if drift_score > 0.1:
                        print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                        data_drift = True

                    if feature_name not in batch_feature_drifts:
                        batch_feature_drifts[feature_name] = []
                    batch_feature_drifts[feature_name].append(drift_score)

        # Calculate the mean drift scores for the chunk and batch
        chunk_drift_mean = np.mean(chunk_drift_scores) if chunk_drift_scores else 0
        batch_drift_mean = np.mean(batch_drift_scores) if batch_drift_scores else 0
        chunk_drift_mean_list.append(chunk_drift_mean)
        batch_drift_mean_list.append(batch_drift_mean)

        if not data_drift:
            print("✅ No significant drift detected.")


        n += 1




    report.run(current_data=test_X.tail(5000),  reference_data=train_X, column_mapping=column_mapping)
    report.save_html("fileSmall.html")
    print("HTML-report saved in:", os.path.abspath("fileSmall.html"))

    report.run(current_data=test_X,  reference_data=train_X, column_mapping=column_mapping)
    report.save_html("file.html")
    print("HTML-report saved in:", os.path.abspath("file.html"))

    result = report.as_dict()

    for metric in result["metrics"]:
        if metric["metric"] == "DataDriftTable":
            drift_by_columns = metric["result"].get("drift_by_columns", {})
            for feature_name, feature_data in drift_by_columns.items():
                drift_score = feature_data["drift_score"]
                if drift_score > 0.1:
                    print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                    data_drift = True




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