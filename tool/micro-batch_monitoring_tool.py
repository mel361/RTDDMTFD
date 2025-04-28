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
from tool.CONSTANT_VALUES import *

# Import precision and recall
metrics_df = pd.read_csv( PATH_METRICS, index_col=0)
precision_list = metrics_df["precision"].tolist()
recall_list = metrics_df["recall"].tolist()
print("Metrics imported")

# Import data
test_X = pd.read_csv(PATH_TEST_X, index_col=0)
test_y = pd.read_csv(PATH_TEST_Y, index_col=0)
train_X = pd.read_csv(PATH_TRAIN_X, index_col=0)
print("Data imported")

# Import threshold
with open(PATH_BEST_THRESHOLD) as f:
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


# Initialize lists and dicts
full_time_list = []
all_iteration_times = {}
micro_batch_drift_values = {}
micro_batch_feature_drift_values = {feature: {} for feature in FRAUD_FEATURES}

for i in range(TEST_ITERATIONS):
    # Set the chunk size
    chunk_size = CHUNK_SIZE
    n = 1
    full_timer_start = time.time()
    time_drift_detected = 0

    # Initialize a list to store iteration times
    iteration_times = []
    per_chunk_drift = []
    per_feature_chunk_drift = {feature: [] for feature in FRAUD_FEATURES}
    # Simulate real-time data drift monitoring
    for j in range(0, len(test_X), chunk_size):
        iteration_time = time.time()
        print("Processing chunk: ", n, "////////////////////////////////////////")
        current_batch = test_X.iloc[j:j + chunk_size]


        # Empty lists to store drift scores
        micro_batch_drift_scores = []

        # Create an Evidently report
        report.run(
            current_data=current_batch,
            reference_data=train_X,
            column_mapping=column_mapping
        )
        result_micro_batch = report.as_dict()

        chunk_drift_scores = []
        data_drift = False


        # Check for data drift in the batch
        print("Checking for drift in current batch")
        for metric in result_micro_batch["metrics"]:
            if metric["metric"] == "DataDriftTable":
                drift_by_columns = metric["result"].get("drift_by_columns", {})
                for feature_name, feature_data in drift_by_columns.items():
                    drift_score = feature_data["drift_score"]
                    chunk_drift_scores.append(drift_score)
                    per_feature_chunk_drift[feature_name].append(drift_score)
                    if drift_score > 0.1:
                        time_drift_detected = time.time() - full_timer_start
                        print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                        data_drift = True


        if not data_drift:
            print("✅ No significant drift detected.")

        print("n: ", n)
        per_chunk_drift.append(np.mean(chunk_drift_scores)) if chunk_drift_scores else 0


        n += 1
        iteration_times.append(time.time() - iteration_time)

    print("Time drift detected: ", time_drift_detected)

    for chunk_id in range(n - 1):
        if chunk_id not in all_iteration_times:
            all_iteration_times[chunk_id] = []
        all_iteration_times[chunk_id].append(iteration_times[chunk_id])

    for chunk_id, drift_val in enumerate(per_chunk_drift):
        if chunk_id not in micro_batch_drift_values:
            micro_batch_drift_values[chunk_id] = []
        micro_batch_drift_values[chunk_id].append(drift_val)

    for feature in FRAUD_FEATURES:
        for chunk_id, drift_val in enumerate(per_feature_chunk_drift[feature]):
            if chunk_id not in micro_batch_feature_drift_values[feature]:
                micro_batch_feature_drift_values[feature][chunk_id] = []
            micro_batch_feature_drift_values[feature][chunk_id].append(drift_val)

    full_time_list.append(time_drift_detected)




# 1. Mean iteration time per chunk
mean_iteration_times = [np.mean(all_iteration_times[chunk_id]) for chunk_id in sorted(all_iteration_times)]

# 2. Mean drift per chunk
mean_drift_per_chunk = [np.mean(micro_batch_drift_values[chunk_id]) for chunk_id in sorted(micro_batch_drift_values)]

# 3. Mean drift per feature per chunk
mean_feature_drift_per_chunk = {
    feature: [np.mean(micro_batch_feature_drift_values[feature][chunk_id])
              for chunk_id in sorted(micro_batch_feature_drift_values[feature])]
    for feature in FRAUD_FEATURES
}

# 4. Mean full drift-tid
mean_full_time = np.mean(full_time_list)
print("Mean time drift detected: ", time_drift_detected)

# --------- Iteration times
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(mean_iteration_times) + 1), mean_iteration_times, marker='o')
plt.title("Iteration times")
plt.xlabel("Iterations")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------- Iteration times
plt.figure(figsize=(10, 4))
plt.plot(range(1, len(full_time_list) + 1), full_time_list, marker='o')
plt.title("Full-time times")
plt.xlabel("tests")
plt.ylabel("Time (seconds)")
plt.grid(True)
plt.tight_layout()
plt.show()


print("micro_batch_drift_mean_list SIZE: ", len(mean_drift_per_chunk))
print("precision_list SIZE: ", len(precision_list))

plt.figure(figsize=(8, 5))
plt.scatter(mean_drift_per_chunk, precision_list, alpha=0.7, label="Precision vs micro-batch Drift")
plt.scatter(mean_drift_per_chunk, recall_list, alpha=0.7, label="Recall vs micro-batch Drift")
plt.xlabel("Mean micro-batch Drift")
plt.ylabel("Score")
plt.title("Precision/Recall vs micro-batch Drift")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




# Print trend for precision per drift score
plt.figure(figsize=(8, 5))
sns.regplot(x=mean_drift_per_chunk, y=precision_list, lowess=True)
plt.xlabel("Mean micro-batch Drift")
plt.ylabel("Precision")
plt.title("Drift vs Precision with Trend line")
plt.grid(True)
plt.tight_layout()
plt.show()

# Print trend for recall per drift score
plt.figure(figsize=(8, 5))
sns.regplot(x=mean_drift_per_chunk, y=recall_list, lowess=True)
plt.xlabel("Mean micro-batch Drift")
plt.ylabel("Recall")
plt.title("Drift vs Recall with Trend line")
plt.grid(True)
plt.tight_layout()
plt.show()

#Create_Graphs.print_graphs(chunk_ids, precision_list, recall_list, chunk_drift_mean_list, batch_drift_mean_list, chunk_feature_drifts, batch_feature_drifts)

Statistic_tests.correlation_test(precision_list, mean_drift_per_chunk, "precision", "micro-batch mean")
Statistic_tests.correlation_test(recall_list, mean_drift_per_chunk, "recall", "micro-batch mean")




# Create graphs for each feature to visualize precision and recall per drift scores
for feature in FRAUD_FEATURES:
    # Print trend for precision per drift score
    plt.figure(figsize=(8, 5))
    sns.regplot(x=mean_feature_drift_per_chunk[feature], y=precision_list, lowess=True)
    plt.xlabel(feature.capitalize() + " micro-batch Drift")
    plt.ylabel("Precision")
    plt.title(feature.capitalize() + ": Drift vs Precision with Trend line")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    Statistic_tests.correlation_test(precision_list, mean_feature_drift_per_chunk[feature], "precision", feature)

    # Print trend for recall per drift score
    plt.figure(figsize=(8, 5))
    sns.regplot(x=mean_feature_drift_per_chunk[feature], y=recall_list, lowess=True)
    plt.xlabel(feature.capitalize() + " micro-batch Drift")
    plt.ylabel("Recall")
    plt.title(feature.capitalize() + ": Drift vs Recall with Trend line")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    Statistic_tests.correlation_test(recall_list, mean_feature_drift_per_chunk[feature], "recall", feature)



pd.DataFrame({
    "Mean drift": mean_drift_per_chunk,
    "Mean iteration time": mean_iteration_times,
    "recall": full_time_list,
    **{feature: drift_list for feature, drift_list in mean_feature_drift_per_chunk.items()}
}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS, index=False)