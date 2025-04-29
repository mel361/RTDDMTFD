# Real-Time Machine Learning Model Monitoring for Banking Fraud Detection: A Micro-Batch Approach with Evidently AI
This project develops and evaluates two monitoring tools for Machine Learning models used in fraud detection, one using **full-batch** approach and one using **micro-batch** approach to approximate **real-time monitoring**.  
The project uses the **Evidently AI** framework and Random Forest models to detect **data drift** in transaction streams.

## Features
- Full-batch and micro-batch drift monitoring of fraud detection models
- Early detection of data drift affecting critical fraud features
- Evaluation of how drift correlates with model precision and recall
- CI/CD pipeline integration for automated deployment
- Statistical correlation analysis between drift and performance degradation

## Contributors
- Rama Bito
- Melker Elofsson

## Installation
Clone the repository:
```bash
git clone https://github.com/https://github.com/mel361/RTDDMTFD
```
## Usage
1. Train the model and find the best threshold:
```bash
python train_model.py
```

2. Run Full Batch Monitoring:
```bash
python full-batch_monitoring.py
```

3. Run Micro Batch Monitoring:
```bash
python micro-batch_monitoring_tool.py
```

Results such as drift statistics, precision/recall scores, and graphs are saved to the /statistics/ directory.

### Requirements
- Python 3.10+
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- evidently
- imbalanced-learn

### System Overview
The tool simulates real-time monitoring using batch and micro-batch processing:

- Full batches analyze accumulated data at each monitoring point.
- Micro-batches analyze smaller windows of recent data.
- Drift detection is performed feature-wise with Evidently AI.
- Performance (precision and recall) is correlated to detected drift over time.
