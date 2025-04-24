import json
import time
import seaborn as sns

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from tool import Statistic_tests

DRIFT_THRESHOLD = 0.1
CHUNK_SIZE = 4000
FRAUD_FEATURES = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count',
                      'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w']

# Import precision and recall
metrics_df = pd.read_csv("metrics.csv", index_col=0)
precision_list = metrics_df["precision"].tolist()
recall_list = metrics_df["recall"].tolist()
print("Metrics imported")

# Import data
test_X = pd.read_csv("test_X.csv", index_col=0)
test_y = pd.read_csv("test_y.csv", index_col=0)
train_X = pd.read_csv("train_X.csv", index_col=0)
print("Data imported")

# Import threshold
with open("best_threshold.json") as f:
    best_threshold = json.load(f)["best_threshold"]
print("Threshold imported")

# Column mapping for Evidently
column_mapping = ColumnMapping(
    numerical_features=FRAUD_FEATURES,
    categorical_features=[],
)
# Create a report with DataDriftPreset
report = Report([
    DataDriftPreset(),
])


# Collect metrics for the graphs
chunk_ids = []
full_batch_drift_mean_list = []

# To keep track of drift scores for each feature
full_batch_feature_drifts = {}

# Set the chunk size
chunk_size = CHUNK_SIZE
n = 1
full_timer_start = time.time()
time_drift_detected = 0

# Initialize a list to store iteration times
iteration_times = []
# Simulate real-time data drift monitoring
for i in range(0, len(test_X), chunk_size):
    iteration_time = time.time()
    print("Processing chunk: ", n, "////////////////////////////////////////")
    current_batch = test_X.iloc[0:i + chunk_size]

    chunk_ids.append(n)

    # Empty lists to store drift scores
    full_batch_drift_scores = []

    # Create an Evidently report
    report.run(
        current_data=current_batch,
        reference_data=train_X,
        column_mapping=column_mapping
    )
    result_full_batch = report.as_dict()

    data_drift = False


    # Check for data drift in the batch
    print("Checking for drift in current batch")
    for metric in result_full_batch["metrics"]:
        if metric["metric"] == "DataDriftTable":
            drift_by_columns = metric["result"].get("drift_by_columns", {})
            for feature_name, feature_data in drift_by_columns.items():
                drift_score = feature_data["drift_score"]
                full_batch_drift_scores.append(drift_score)
                if drift_score > 0.1:
                    time_drift_detected = time.time() - full_timer_start
                    print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                    data_drift = True

                if feature_name not in full_batch_feature_drifts:
                    full_batch_feature_drifts[feature_name] = []
                full_batch_feature_drifts[feature_name].append(drift_score)

    # Calculate the mean drift scores for full_batch
    full_batch_drift_mean = np.mean(full_batch_drift_scores) if full_batch_drift_scores else 0
    full_batch_drift_mean_list.append(full_batch_drift_mean)

    if not data_drift:
        print("✅ No significant drift detected.")


    n += 1
    iteration_times.append(time.time() - iteration_time)






print("Time drift detected: ", time_drift_detected)

# --------- Iteration times
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(iteration_times) + 1), iteration_times, marker='o')
plt.title("Iteration times")
plt.xlabel("Iterations")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.show()




plt.figure(figsize=(8, 5))
plt.scatter(full_batch_drift_mean_list, precision_list, alpha=0.7, label="Precision vs full-batch Drift")
plt.scatter(full_batch_drift_mean_list, recall_list, alpha=0.7, label="Recall vs full-batch Drift")
plt.xlabel("Mean full-batch Drift")
plt.ylabel("Score")
plt.title("Precision/Recall vs full-batch Drift")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Print trend for precision per drift score
plt.figure(figsize=(8, 5))
sns.regplot(x=full_batch_drift_mean_list, y=precision_list, lowess=True)
plt.xlabel("Mean Full-batch Drift")
plt.ylabel("Precision")
plt.title("Drift vs Precision with Trend line")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print trend for recall per drift score
plt.figure(figsize=(8, 5))
sns.regplot(x=full_batch_drift_mean_list, y=recall_list, lowess=True)
plt.xlabel("Mean Full-batch Drift")
plt.ylabel("Recall")
plt.title("Drift vs Recall with Trend line")
plt.grid(True)
plt.tight_layout()
plt.show()

#Create_Graphs.print_graphs(chunk_ids, precision_list, recall_list, chunk_drift_mean_list, batch_drift_mean_list, chunk_feature_drifts, batch_feature_drifts)

Statistic_tests.correlation_precision_test(precision_list, full_batch_drift_mean_list)
Statistic_tests.correlation_recall_test(recall_list, full_batch_drift_mean_list)

