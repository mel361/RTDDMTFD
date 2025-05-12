import os

from CONSTANT_VALUES import *

# Create directories for storing output graphs and statistics
dirs_to_create = [
    os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'micro_batch_tool_statistics'),
    os.path.join(script_dir, '..', VARIANT_NAME, 'statistics', 'full_batch_tool_statistics'),
    os.path.join(PATH_OUTPUT_GRAPHS_COMPARISON),
    os.path.join(PATH_OUTPUT_MICRO_COMPARISON),
    os.path.join(PATH_OUTPUT_FULL_COMPARISON),
]


# Create the directories if they do not exist
for path in dirs_to_create:
    os.makedirs(path, exist_ok=True)