import os


# Constants for the monitoring tool

VARIANT_NAME = os.getenv("VARIANT_NAME", "NOT DEFINED") # The Name of the variant being monitored. This can be set as an environment variable.

# Number of rows to process in each chunk during batch monitoring.
CHUNK_SIZE = 4000

# Amount of data to be used for testing. This is the size of the test dataset for finding best f1-score
TEST_SIZE = 100000

# Number of tests for each monitoring tool. This is used to simulate multiple runs of the monitoring process.
TEST_ITERATIONS = 1

# List of features used for fraud detection. Drift will be monitored for these features.
FRAUD_FEATURES = [
    'income',
    'name_email_similarity',
    'prev_address_months_count',
    'current_address_months_count',
    'customer_age',
    'days_since_request',
    'intended_balcon_amount',
    'zip_count_4w'
]

# File paths for input and output data used in the monitoring tool.
PATH_TRAIN_X = "model_values\\train_X.csv"  # Path to the training dataset (features only).
PATH_TEST_Y = "model_values\\test_y.csv"   # Path to the test dataset (labels only).
PATH_TEST_X = "model_values\\test_X.csv"   # Path to the test dataset (features only).
PATH_METRICS = "model_values\\metrics.csv" # Path to save evaluation metrics.
PATH_BEST_THRESHOLD = "model_values\\best_threshold.json" # Path to save the best threshold values.
PATH_FULL_BATCH_TOOL_STATISTICS_MEAN_DRIFTS = f"..\\{VARIANT_NAME}\\statistics\\full_batch_tool_statistics\\mean_drifts.csv"
PATH_MICRO_BATCH_TOOL_STATISTICS_MEAN_DRIFTS = f"..\\{VARIANT_NAME}\\statistics\\micro_batch_tool_statistics\\mean_drifts.csv"
PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES = f"..\\{VARIANT_NAME}\\statistics\\full_batch_tool_statistics\\features.csv"
PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES = f"..\\{VARIANT_NAME}\\statistics\\micro_batch_tool_statistics\\features.csv"
PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_TIMES = f"..\\{VARIANT_NAME}\\statistics\\full_batch_tool_statistics\\detection_times.csv"
PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_TIMES = f"..\\{VARIANT_NAME}\\statistics\\micro_batch_tool_statistics\\detection_times.csv"
PATH_FULL_BATCH_TOOL_STATISTICS_ITERATION_TIMES = f"..\\{VARIANT_NAME}\\statistics\\full_batch_tool_statistics\\iteration_times.csv"
PATH_MICRO_BATCH_TOOL_STATISTICS_ITERATION_TIMES = f"..\\{VARIANT_NAME}\\statistics\\micro_batch_tool_statistics\\iteration_times.csv"
PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_IDS = f"..\\{VARIANT_NAME}\\statistics\\full_batch_tool_statistics\\detection_ids.csv"
PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_IDS = f"..\\{VARIANT_NAME}\\statistics\\micro_batch_tool_statistics\\detection_ids.csv"


# Paths for micro output graphs to be stored.
PATH_MICRO = f"..\\{VARIANT_NAME}\\output_graphs\\micro-batch\\"
PATH_GRAPH_MICRO_ITERATIONS = PATH_MICRO + "iteration_times.png"
PATH_GRAPH_MICRO_FULL_TIME = PATH_MICRO + "full_time.png"
PATH_GRAPH_MICRO_PRECISION_TREND = PATH_MICRO + "drift_vs_precision.png"
PATH_GRAPH_MICRO_RECALL_TREND = PATH_MICRO + "drift_vs_recall.png"
PATH_GRAPH_MICRO_FEATURES_PRECISION = "_drift_vs_precision.png"
PATH_GRAPH_MICRO_FEATURES_RECALL = "_drift_vs_recall.png"

# Paths for full output graphs to be stored.
PATH_FULL = f"..\\{VARIANT_NAME}\\output_graphs\\full-batch\\"
PATH_GRAPH_FULL_ITERATIONS = PATH_FULL + "iteration_times.png"
PATH_GRAPH_FULL_FULL_TIME = PATH_FULL + "full_time.png"
PATH_GRAPH_FULL_PRECISION_TREND = PATH_FULL + "drift_vs_precision.png"
PATH_GRAPH_FULL_RECALL_TREND = PATH_FULL + "drift_vs_recall.png"
PATH_GRAPH_FULL_FEATURES_PRECISION = "_drift_vs_precision.png"
PATH_GRAPH_FULL_FEATURES_RECALL = "_drift_vs_recall.png"



