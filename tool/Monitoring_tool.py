import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import Create_Graphs
import Statistic_tests
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from imblearn.over_sampling import SMOTE
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import f1_score, recall_score, precision_score


DRIFT_THRESHOLD = 0.1

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

# Split the data into training and testing sets
train_size = int(len(X) * (0.8))
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



df_drift = pd.DataFrame({
    "recall": recall_list,
    "precision": precision_list
})

recall_deltas = []
precision_deltas = []
labels = []

for feature in fraud_features:
    drift_col_name = feature + "_drifted"
    drift_status = np.array(chunk_feature_drifts[feature]) > DRIFT_THRESHOLD
    df_drift[drift_col_name] = drift_status

for feature in fraud_features:
    drift_col = feature + "_drifted"
    recall_mean_with_drift = df_drift[df_drift[drift_col]]["recall"].mean()
    recall_mean_without_drift = df_drift[~df_drift[drift_col]]["recall"].mean()
    delta = recall_mean_with_drift - recall_mean_without_drift
    print(f"{feature}: Δ Recall = {delta:.4f} (With drift: {recall_mean_with_drift:.4f}, Without drift: {recall_mean_without_drift:.4f})")
    precision_mean_with_drift = df_drift[df_drift[drift_col]]["precision"].mean()
    precision_mean_without_drift = df_drift[~df_drift[drift_col]]["precision"].mean()
    delta = precision_mean_with_drift - precision_mean_without_drift
    print(f"{feature}: Δ Precision = {delta:.4f} (With drift: {precision_mean_with_drift:.4f}, Without drift: {precision_mean_without_drift:.4f})")

    recall_deltas.append([recall_mean_with_drift, recall_mean_without_drift])
    precision_deltas.append([precision_mean_with_drift, precision_mean_without_drift])
    labels.append(feature)



# --------- Recall
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(12, 5))
plt.bar(x - width/2, [val[0] for val in recall_deltas], width, label='Recall with Drift')
plt.bar(x + width/2, [val[1] for val in recall_deltas], width, label='Recall without Drift')
plt.ylabel('Recall')
plt.title('Recall med vs utan drift per feature')
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# --------- Precision
plt.figure(figsize=(12, 5))
plt.bar(x - width/2, [val[0] for val in precision_deltas], width, label='Precision with Drift')
plt.bar(x + width/2, [val[1] for val in precision_deltas], width, label='Precision without Drift')
plt.ylabel('Precision')
plt.title('Precision med vs utan drift per feature')
plt.xticks(x, labels, rotation=45)
plt.legend()
plt.tight_layout()
plt.show()







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




#Create_Graphs.print_graphs(chunk_ids, precision_list, recall_list, chunk_drift_mean_list, batch_drift_mean_list, chunk_feature_drifts, batch_feature_drifts)

#Statistic_tests.run_statistic_tests(chunk_drift_mean_list, batch_drift_mean_list, precision_list, recall_list)

