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

# Create a report with DataDriftPreset
report = Report([
    DataDriftPreset(),
])

# Initialize lists and dicts
drift_detection_time = []
drift_start_ids = []
all_iteration_times = {}
micro_batch_drift_values = {}
micro_batch_feature_drift_values = {feature: {} for feature in FRAUD_FEATURES}
for i in range(TEST_ITERATIONS):
    # Set the chunk size
    chunk_size = CHUNK_SIZE
    n = 1
    print("Test: ", i)
    full_timer_start = time.time()
    time_drift_detected = 0
    total_features_drifting_list = []
    drift_start_id = 0
    # Initialize a list to store iteration times
    iteration_times = []
    per_chunk_drift = []
    per_feature_chunk_drift = {feature: [] for feature in FRAUD_FEATURES}

    data_drift = False
    # Simulate real-time data drift monitoring
    for j in range(0, len(test_X), chunk_size):
        iteration_time = time.time()
        print("Processing chunk: ", n, "////////////////////////////////////////")
        current_batch = test_X.iloc[j:j + chunk_size]
        features_drifting = 0

        # Empty lists to store drift scores
        micro_batch_drift_scores = []

        # filter out constant columns
        constant_cols = current_batch.columns[current_batch.nunique() <= 2].tolist()
        variable_cols = current_batch.columns[current_batch.nunique() > 2].tolist()
        print("Constant columns: ", constant_cols)
        print("Variable columns: ", variable_cols)
        # Check if the current batch is empty
        filtered_batch = current_batch[variable_cols]
        filtered_train_X = train_X[variable_cols]
        for col in constant_cols:
            print("\nFiltered batch  ", col, ": ",  current_batch[col].unique())
            print("\nFiltered train  ", col, ": ",  current_batch[col].unique())
        # Column mapping for Evidently
        filtered_column_mapping = ColumnMapping(
            numerical_features=[col for col in variable_cols],
            categorical_features=[]
        )

        # Create an Evidently report
        report.run(
            current_data=filtered_batch,
            reference_data=filtered_train_X,
            column_mapping=filtered_column_mapping
        )
        # Get the report
        result_micro_batch = report.as_dict()

        chunk_drift_scores = []

        print(FRAUD_FEATURES)
        # set constant drift to 0
        for const_feature in constant_cols:
            print(f"Constant feature: {const_feature}")
            chunk_drift_scores.append(0.0)
            per_feature_chunk_drift[const_feature].append(0.0)

        # Check for data drift in the batch
        print("Checking for drift in current batch")
        for metric in result_micro_batch["metrics"]:
            if metric["metric"] == "DataDriftTable":
                drift_by_columns = metric["result"].get("drift_by_columns", {})
                for feature_name, feature_data in drift_by_columns.items():
                    drift_score = feature_data["drift_score"]
                    chunk_drift_scores.append(drift_score)
                    per_feature_chunk_drift[feature_name].append(drift_score)
                    threshold = feature_data["stattest_threshold"]
                    if threshold is not None:
                        if drift_score > threshold:
                            print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f} > threshold {threshold:.3f}")
                            features_drifting += 1
                            if not data_drift:
                                if features_drifting / len(FRAUD_FEATURES) > 0.5:
                                    time_drift_detected = time.time() - full_timer_start
                                    drift_start_id = n
                                    data_drift = True


        if not data_drift:
            print("✅ No significant drift detected.")


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
        if chunk_id not in micro_batch_drift_values:
            micro_batch_drift_values[chunk_id] = []
        micro_batch_drift_values[chunk_id].append(drift_val)

    for feature in FRAUD_FEATURES:
        for chunk_id, drift_val in enumerate(per_feature_chunk_drift[feature]):
            if chunk_id not in micro_batch_feature_drift_values[feature]:
                micro_batch_feature_drift_values[feature][chunk_id] = []
            micro_batch_feature_drift_values[feature][chunk_id].append(drift_val)

    drift_detection_time.append(time_drift_detected)
    drift_start_ids.append(drift_start_id)





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

# Correlation tests
statistic_test_results = {"Name": [], "Precision test name": [], "Precision p-value": [], "Precision significant?": [], "Precision correlation": [],
                          "Precision correlation type": [],"Recall test name" : [], "Recall p-value": [], "Recall significant?": [], "Recall correlation": [],
                          "Recall correlation type": []}

statistic_test_results["Name"].append("Micro-batch mean")
corr, pval, test_name = statistic_tests.correlation_test(precision_list, mean_drift_per_chunk, "precision", "micro-batch mean")
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

corr, pval, test_name = statistic_tests.correlation_test(recall_list, mean_drift_per_chunk, "recall", "micro-batch mean")
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
    corr, pval, test_name = statistic_tests.correlation_test(precision_list, mean_feature_drift_per_chunk[feature], "precision", f"micro-batch {feature}")
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

    corr, pval, test_name = statistic_tests.correlation_test(recall_list, mean_feature_drift_per_chunk[feature], "recall",f"micro-batch {feature}")
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


statistic_test_results["Name"].append("Micro-batch features_drifting")
corr, pval, test_name = statistic_tests.correlation_test(precision_list, total_features_drifting_list, "precision", "micro-batch features_drifting")
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


corr, pval, test_name = statistic_tests.correlation_test(recall_list, total_features_drifting_list, "recall", "micro-batch features_drifting")
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
}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES, index=False)

# Save the drift values for each chunk
pd.DataFrame({"Mean drift": mean_drift_per_chunk}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_MEAN_DRIFTS, index=False)

# Save drift detection times
pd.DataFrame({"Drift detection times": drift_detection_time}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_TIMES, index=False)

# Save iteration times
pd.DataFrame({"Iteration times": mean_iteration_times}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_ITERATION_TIMES, index=False)

# Save drift detection ids
pd.DataFrame({"Drift detection ids": drift_start_ids}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_IDS, index=False)

# Save the number of features drifting per chunk
pd.DataFrame({"Features drifting": total_features_drifting_list}).to_csv(PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES_DRIFTING, index=False)


for key, val in statistic_test_results.items():
    print(f"{key}: {len(val)}")
# Save the statistic test results
statistic_test_results_df = pd.DataFrame(statistic_test_results)
statistic_test_results_df.to_csv(PATH_STATISTIC_TESTS_MICRO, index=False)