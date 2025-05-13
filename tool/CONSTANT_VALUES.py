import os
script_dir = os.path.dirname(os.path.abspath(__file__))

# Constants for the monitoring tool

VARIANT_NAME = os.getenv("VARIANT_NAME", "NOT DEFINED") # The Name of the variant being monitored. This can be set as an environment variable.

# Number of rows to process in each chunk during batch monitoring.
CHUNK_SIZE = 8000

# Amount of data to be used for testing. This is the size of the test dataset for finding best f1-score
TEST_SIZE = 100000

# Number of tests for each monitoring tool. This is used to simulate multiple runs of the monitoring process.
TEST_ITERATIONS = 1

# List of features used for fraud detection. Drift will be monitored for these features.
FRAUD_FEATURES = [
    "customer_age",
    "current_address_months_count",
    "keep_alive_session",
    "phone_home_valid",
    "device_distinct_emails_8w",
    "intended_balcon_amount",
    "zip_count_4w",
    "velocity_24h",
    "bank_months_count"
]


PATH_REFERENCE_DATASET =  os.path.join(script_dir, '..', 'data', 'Reference.csv')# Path to the reference dataset used for training the model.
PATH_NEW_DATASET =  os.path.join(script_dir, '..', 'data', 'NewData.csv') # Path to the new dataset used for testing the model.

# File paths for input and output data used in the monitoring tool.
PATH_TRAIN_X = os.path.join(script_dir, 'model_values', 'train_X.csv')
PATH_TEST_Y = os.path.join(script_dir, 'model_values', 'test_y.csv')
PATH_TEST_X = os.path.join(script_dir, 'model_values', 'test_X.csv')
PATH_METRICS = os.path.join(script_dir, 'model_values', 'metrics.csv')
PATH_BEST_THRESHOLD = os.path.join(script_dir, 'model_values', 'best_threshold.json')

PATH_FULL_BATCH_TOOL_STATISTICS_MEAN_DRIFTS = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics', 'mean_drifts.csv')
PATH_MICRO_BATCH_TOOL_STATISTICS_MEAN_DRIFTS = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics', 'mean_drifts.csv')
PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics', 'features.csv')
PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics', 'features.csv')
PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_TIMES = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics', 'detection_times.csv')
PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_TIMES = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics', 'detection_times.csv')
PATH_FULL_BATCH_TOOL_STATISTICS_ITERATION_TIMES = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics', 'iteration_times.csv')
PATH_MICRO_BATCH_TOOL_STATISTICS_ITERATION_TIMES = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics', 'iteration_times.csv')
PATH_FULL_BATCH_TOOL_STATISTICS_DETECTION_IDS = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics', 'detection_ids.csv')
PATH_MICRO_BATCH_TOOL_STATISTICS_DETECTION_IDS = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics', 'detection_ids.csv')
PATH_FULL_BATCH_TOOL_STATISTICS_FEATURES_DRIFTING = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics', 'features_drifting.csv')
PATH_MICRO_BATCH_TOOL_STATISTICS_FEATURES_DRIFTING = os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics', 'features_drifting.csv')

# Create directories for storing output graphs and statistics
PATH_OUTPUT_GRAPHS_COMPARISON = os.path.join(script_dir, '..', VARIANT_NAME, 'comparison', 'graphs')
PATH_OUTPUT_MICRO_COMPARISON = os.path.join(script_dir, '..', VARIANT_NAME, 'comparison', 'micro_batch_tool_statistics')
PATH_OUTPUT_FULL_COMPARISON = os.path.join(script_dir, '..', VARIANT_NAME, 'comparison', 'full_batch_tool_statistics')

# Path to the statistic tests csv
PATH_STATISTIC_TESTS_MICRO = os.path.join(PATH_OUTPUT_MICRO_COMPARISON, "statistic_tests.csv")
PATH_STATISTIC_TESTS_FULL = os.path.join(PATH_OUTPUT_FULL_COMPARISON, "statistic_tests.csv")



