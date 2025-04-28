# Constants for the monitoring tool

# Threshold for detecting data drift. If the drift score exceeds this value, drift is flagged.
DRIFT_THRESHOLD = 0.1

# Number of rows to process in each chunk during batch monitoring.
CHUNK_SIZE = 4000

# Amount of data to be used for testing. This is the size of the test dataset for finding best f1-score
TEST_SIZE = 40000

# Number of tests for each monitoring tool. This is used to simulate multiple runs of the monitoring process.
TEST_ITERATIONS = 3

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
PATH_TRAIN_X = "model_values/train_X.csv"  # Path to the training dataset (features only).
PATH_TEST_Y = "model_values/test_y.csv"   # Path to the test dataset (labels only).
PATH_TEST_X = "model_values/test_X.csv"   # Path to the test dataset (features only).
PATH_METRICS = "model_values/metrics.csv" # Path to save evaluation metrics.
PATH_BEST_THRESHOLD = "model_values/best_threshold.json" # Path to save the best threshold values.
PATH_FULL_BATCH_TOOL_STATISTICS = "../statistics/full_batch_tool_statistics.csv" # Path to save statistics from the full batch monitoring tool.
PATH_MICRO_BATCH_TOOL_STATISTICS = "../statistics/micro_batch_tool_statistics.csv" # Path to save statistics from the micro batch monitoring tool.