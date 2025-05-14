import json
import time

import numpy as np
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

import statistic_tests
from CONSTANT_VALUES import *

# Import precision and recall
metrics_df = pd.read_csv(PATH_METRICS, index_col=0)
precision_list = metrics_df["precision"].tolist()
recall_list = metrics_df["recall"].tolist()
print("Metrics imported")

# Import data
test_X = pd.read_csv(PATH_TEST_X, index_col=0)
test_y = pd.read_csv(PATH_TEST_Y, index_col=0)
train_X = pd.read_csv(PATH_TRAIN_X, index_col=0)
print("Data imported")


# Import threshold
with open(PATH_MODEL_CONFIG) as f:
    config = json.load(f)
    best_threshold = config["best_threshold"]
    FRAUD_FEATURES = config["fraud_features"]
print("Threshold imported and features loaded")

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
drift_detection_time = []
drift_start_ids = []
all_iteration_times = {}
full_batch_drift_values = {}
full_batch_feature_drift_values = {feature: {} for feature in FRAUD_FEATURES}

for i in range(TEST_ITERATIONS):
    # Set the chunk size
    chunk_size = CHUNK_SIZE
    n = 1
    print("Test: ", i)
    full_timer_start = time.time()
    time_drift_detected = 0

    drift_start_id = 0

    total_features_drifting_list = []
    # Initialize a list to store iteration times'
    iteration_times = []
    per_chunk_drift = []
    per_feature_chunk_drift = {feature: [] for feature in FRAUD_FEATURES}
    # Simulate real-time data drift monitoring
    for j in range(0, len(test_X), chunk_size):
        iteration_time = time.time()
        print("Processing chunk: ", n, "////////////////////////////////////////")
        current_batch = test_X.iloc[0:j + chunk_size]
        features_drifting = 0

        # Empty lists to store drift scores
        full_batch_drift_scores = []

        # Create an Evidently report
        report.run(
            current_data=current_batch,
            reference_data=train_X,
            column_mapping=column_mapping
        )
        result_full_batch = report.as_dict()

        chunk_drift_scores = []
        data_drift = False


        # Check for data drift in the batch
        print("Checking for drift in current batch")
        for metric in result_full_batch["metrics"]:
            if metric["metric"] == "DataDriftTable":
                drift_by_columns = metric["result"].get("drift_by_columns", {})
                for feature_name, feature_data in drift_by_columns.items():
                    drift_score = feature_data["drift_score"]
                    chunk_drift_scores.append(drift_score)
                    per_feature_chunk_drift[feature_name].append(drift_score)
                    threshold = feature_data["stattest_threshold"]
                    if threshold is not None:
                        if drift_score > threshold:
                            time_drift_detected = time.time() - full_timer_start
                            print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f} > threshold {threshold:.3f}")
                            features_drifting += 1
                            data_drift = True


        if not data_drift:
            print("✅ No significant drift detected.")

        print("n: ", n)
        per_chunk_drift.append(np.mean(chunk_drift_scores)) if chunk_drift_scores else 0


        n += 1
        iteration_times.append(time.time() - iteration_time)
        total_features_drifting_list.append(features_drifting)


    print("Time drift detected: ", time_drift_detected)

    for chunk_id in range(n - 1):
        if chunk_id not in all_iteration_times:
            all_iteration_times[chunk_id] = []
        all_iteration_times[chunk_id].append(iteration_times[chunk_id])

    for chunk_id, drift_val in enumerate(per_chunk_drift):
        if chunk_id not in full_batch_drift_values:
            full_batch_drift_values[chunk_id] = []
        full_batch_drift_values[chunk_id].append(drift_val)

    for feature in FRAUD_FEATURES:
        for chunk_id, drift_val in enumerate(per_feature_chunk_drift[feature]):
            if chunk_id not in full_batch_feature_drift_values[feature]:
                full_batch_feature_drift_values[feature][chunk_id] = []
            full_batch_feature_drift_values[feature][chunk_id].append(drift_val)

    drift_detection_time.append(time_drift_detected)
    drift_start_ids.append(drift_start_id)




# 1. Mean iteration time per chunk
mean_iteration_times = [np.mean(all_iteration_times[chunk_id]) for chunk_id in sorted(all_iteration_times)]

# 2. Mean drift per chunk
mean_drift_per_chunk = [np.mean(full_batch_drift_values[chunk_id]) for chunk_id in sorted(full_batch_drift_values)]

# 3. Mean drift per feature per chunk
mean_feature_drift_per_chunk = {
    feature: [np.mean(full_batch_feature_drift_values[feature][chunk_id])
              for chunk_id in sorted(full_batch_feature_drift_values[feature])]
    for feature in FRAUD_FEATURES
}


# Correlation tests
statistic_test_results = {"Name": [], "Precision test name": [], "Precision p-value": [], "Precision significant?": [], "Precision correlation": [],
                          "Precision correlation type": [],"Recall test name" : [], "Recall p-value": [], "Recall significant?": [], "Recall correlation": [],
                          "Recall correlation type": []}

statistic_test_results["Name"].append("full-batch mean")
corr, pval, test_name = statistic_tests.correlation_test(precision_list, mean_drift_per_chunk, "precision", "full-batch mean")
statistic_test_results["Precision p-value"].append(pval)
statistic_test_results["Precision correlation"].append(corr)
statistic_test_results["Precision test name"].append(test_name)
if pval < 0.05:
    statistic_test_results["Precision significant?"].append("Yes")
    if corr > 0:
        statistic_test_results["Precision correlation type"].append("Positive")
    else:
        statistic_test_results["Precision correlation type"].append("Negative")
else:
    statistic_test_results["Precision significant?"].append("No")
    statistic_test_results["Precision correlation type"].append("Not significant")

corr, pval, test_name = statistic_tests.correlation_test(recall_list, mean_drift_per_chunk, "recall", "full-batch mean")
statistic_test_results["Recall p-value"].append(pval)
statistic_test_results["Recall correlation"].append(corr)
statistic_test_results["Recall test name"].append(test_name)
if pval < 0.05:
    statistic_test_results["Recall significant?"].append("Yes")
    if corr > 0:
        statistic_test_results["Recall correlation type"].append("Positive")
    else:
        statistic_test_results["Recall correlation type"].append("Negative")
else:
    statistic_test_results["Recall significant?"].append("No")
    statistic_test_results["Recall correlation type"].append("Not significant")



for feature in FRAUD_FEATURES:
    statistic_test_results["Name"].append(feature)
    corr, pval, test_name = statistic_tests.correlation_test(precision_list, mean_feature_drift_per_chunk[feature], "precision", f"full-batch {feature}")
    statistic_test_results["Precision p-value"].append(pval)
    statistic_test_results["Precision correlation"].append(corr)
    statistic_test_results["Precision test name"].append(test_name)
    if pval < 0.05:
        statistic_test_results["Precision significant?"].append("Yes")
        if corr > 0:
            statistic_test_results["Precision correlation type"].append("Positive")
        else:
            statistic_test_results["Precision correlation type"].append("Negative")
    else:
        statistic_test_results["Precision significant?"].append("No")
        statistic_test_results["Precision correlation type"].append("Not significant")

    corr, pval, test_name = statistic_tests.correlation_test(recall_list, mean_feature_drift_per_chunk[feature], "recall",f"full-batch {feature}")
    statistic_test_results["Recall p-value"].append(pval)
    statistic_test_results["Recall correlation"].append(corr)
    statistic_test_results["Recall test name"].append(test_name)
    if pval < 0.05:
        statistic_test_results["Recall significant?"].append("Yes")
        if corr > 0:
            statistic_test_results["Recall correlation type"].append("Positive")
        else:
            statistic_test_results["Recall correlation type"].append("Negative")
    else:
        statistic_test_results["Recall significant?"].append("No")
        statistic_test_results["Recall correlation type"].append("Not significant")



# Save the drift values for each feature
pd.DataFrame({
    feature: drift_list for feature, drift_list in mean_feature_drift_per_chunk.items()
}).to_csv(PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES, index=False)

# Save the drift values for each chunk
pd.DataFrame({"Mean drift": mean_drift_per_chunk}).to_csv(PATH_FULL_BATCH_TOOL_STATISTICS_MEAN_DRIFTS, index=False)

# Save the drift values for each feature
pd.DataFrame({"Drift detection times": drift_detection_time}).to_csv(PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_TIMES, index=False)

# Save iteration times
pd.DataFrame({"Iteration times": mean_iteration_times}).to_csv(PATH_FULL_BATCH_TOOL_STATISTICS_ITERATION_TIMES, index=False)

# Save drift detection ids
pd.DataFrame({"Drift detection ids": drift_start_ids}).to_csv(PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_IDS, index=False)

# Save the number of features drifting per chunk
pd.DataFrame({"Features drifting": total_features_drifting_list}).to_csv(PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES_DRIFTING, index=False)

# Save the statistic test results
statistic_test_results_df = pd.DataFrame(statistic_test_results)
statistic_test_results_df.to_csv(PATH_STATISTIC_TESTS_FULL, index=False)