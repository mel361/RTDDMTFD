import os

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
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
fraud_data = pd.read_csv('../data/FiFAR/Base.csv').head(100000)
fraud_features = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w']


# Split the data into features and target variable
y = fraud_data['fraud_bool']
X = fraud_data[fraud_features]

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)
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

# Predict on the test set
probabilities = model.predict_proba(test_X)[:, 1]
thresholds = np.arange(0.000, 1.000, 0.001)

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
print("-" * 40)

best_percision = (0, 0, 0, 0)
best_f1Score = (0, 0, 0, 0)
best_recall = (0, 0, 0, 0)
for threshold in thresholds:
    preds = (probabilities > threshold).astype(int)
    precision = precision_score(test_y, preds, zero_division=0)
    recall = recall_score(test_y, preds, zero_division=0)
    f1 = f1_score(test_y, preds, zero_division=0)
    if f1 > best_f1Score[3]: best_f1Score = (threshold, precision, recall, f1)
    if precision > best_percision[1]: best_percision = (threshold, precision, recall, f1)
    if recall > best_recall[2]: best_recall = (threshold, precision, recall, f1)
    print(f"{threshold:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

# Print the best thresholds
print("\nBest F1 Score: ", best_f1Score[0], "; With Precision: ", best_f1Score[1], ";  Recall: ", best_f1Score[2], ";  F1: ", best_f1Score[3])
print("Best Precision: ", best_percision[0], "; With Precision: ", best_percision[1], ";  Recall: ", best_percision[2], ";  F1: ", best_percision[3])
print("Best Recall: ", best_recall[0], "; With Precision: ", best_recall[1], ";  Recall: ", best_recall[2], ";  F1: ", best_recall[3])

# Column mapping for Evidently
column_mapping = ColumnMapping(
    numerical_features=fraud_features,
    categorical_features=[],
)
# Create a report with DataDriftPreset
report = Report([
    DataDriftPreset(),
])

# Set the chunk size
chunk_size = 4000
n = 0
# Simulate real-time data drift monitoring
for i in range(0, len(test_X), chunk_size):
    print("Processing chunk: ", n, "////////////////////////////////////////")
    current_chunk = test_X.iloc[i:i + chunk_size]
    current_chunk_target = test_y.iloc[i:i + chunk_size]
    current_batch = test_X.iloc[0:i + chunk_size]

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
                if drift_score > 0.1:
                    print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                    data_drift = True


    # Check for data drift in the batch
    print("Checking for drift in batch")
    for metric in resultBatch["metrics"]:
        if metric["metric"] == "DataDriftTable":
            drift_by_columns = metric["result"].get("drift_by_columns", {})
            for feature_name, feature_data in drift_by_columns.items():
                drift_score = feature_data["drift_score"]
                if drift_score > 0.1:
                    print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                    data_drift = True

    if not data_drift:
        print("✅ No significant drift detected.")


    prediction = model.predict(current_chunk)
    print("Accuracy: " + str(accuracy_score(current_chunk_target, prediction)))
    print("\nClassification Report:\n", classification_report(current_chunk_target, prediction))

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