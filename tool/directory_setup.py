import os

from tool.CONSTANT_VALUES import VARIANT_NAME

# Create directories for storing output graphs and statistics
dirs_to_create = [
    f"../{VARIANT_NAME}/output_graphs/micro-batch/",
    f"../{VARIANT_NAME}/output_graphs/full-batch/",
    f"../{VARIANT_NAME}/statistics/micro_batch_tool_statistics/",
    f"../{VARIANT_NAME}/statistics/full_batch_tool_statistics/",
]

# Create the directories if they do not exist
for path in dirs_to_create:
    os.makedirs(path, exist_ok=True)