# Micro Batch Tool Graphs

This folder contains visualizations related to monitoring data in micro-batches. The tool is designed to analyze and evaluate performance, data drift, and other metrics in smaller data partitions.

## Contents

- **Plots**: Visualizations saved in the `micro-batch` subdirectory under `output_graphs`. These include:
  - `iteration_times.png`: Mean iteration times per chunk.
  - `full_time.png`: Total test times for all iterations.
  - `micro-batch-drift.png`: Precision and recall vs. micro-batch drift.
  - `drift_vs_precision.png`: Drift vs. precision with trend lines.
  - `drift_vs_recall.png`: Drift vs. recall with trend lines.
  - `{feature}_drift_vs_precision.png`: Drift vs. precision for individual features.
  - `{feature}_drift_vs_recall.png`: Drift vs. recall for individual features.

## Purpose

The purpose of this folder is to collect and organize visualizations from micro-batch monitoring. This enables detailed analysis of model performance, detection of data drift, and identification of potential issues in data processing.
