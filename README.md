# Real-Time Machine Learning Model Monitoring for Banking Fraud Detection: A Micro-Batch Approach with Evidently AI
This project develops and evaluates two monitoring tools for Machine Learning models used in fraud detection, one using **full-batch** approach and one using **micro-batch** approach to approximate **real-time monitoring**.  
The project uses the **Evidently AI** framework and Random Forest models to detect **data drift** in transaction streams.

# Project information

## Features
- Full-batch and micro-batch drift monitoring of fraud detection models
- Early detection of data drift affecting critical fraud features
- Evaluation of how drift correlates with model precision and recall
- CI/CD pipeline integration for automated deployment
- Statistical correlation analysis between drift and performance degradation

## Contributors
- Rama Bito
- Melker Elofsson

## Continuous Integration (CI)
This project uses **GitHub Actions** to automate testing and monitoring workflows.
The pipeline does the following:

- Sets up a clean virutal Ubuntu enviroment
- Sets up python
- Installs dependencies
- Downloads the dataset
- Runs directory setup script
- Runs the training script
- Runs both micro-batch and full-batch monitoring tools
- Runs script for result comparison

The workflow is triggered on:

- Push or pull request events to the feature_drift_correlation branch
- Manual trigger via the **GitHub Actions** tab

Workflow file: .github/workflows/variant.yml

## System Overview
The tool simulates real-time monitoring using batch and micro-batch processing:

- Full batches analyze accumulated data at each monitoring point.
- Micro-batches analyze smaller windows of recent data.
- Drift detection is performed feature-wise with Evidently AI.
- Performance (precision and recall) is correlated to overall detected drift over time aswell as feature-wise.
- Results är visualized by graphs.

# Reproduction instructions

## Installation
Clone the repository:
```bash
git clone https://github.com/mel361/RTDDMTFD
```

Move to directory
```bash
cd RTDDMTFD
```

Intall dependencies
```bash
pip install -r requirements.txt
```
## Usage
1. Download Reference dataset. 
```bash
curl -L -o Base.zip https://github.com/mel361/RTDDMTFD/releases/download/v1.0-data/Base.zip
7z x Base.zip -odata
move data\Base.csv data\Reference.csv
```

2. Download Variant dataset. NOTE! Replace <VARIANT_NAME> with the name of the variant. Existing variants in release v1.0-data are:
- VariantI
- VariantII
- VariantIII
- VariantIV
- VariantV
```bash
curl -L -o <VARIANT_NAME>.zip https://github.com/mel361/RTDDMTFD/releases/download/v1.0-data/<VARIANT_NAME>.zip
7z x <VARIANT_NAME>.zip -odata
move data\<VARIANT_NAME>.csv data\NewData.csv
```

3. Move to tool directory
```bash
cd tool
```

4. Set enviroment variable to the wanted Variant name as in step 2
```bash
  set VARIANT_NAME=<VARIANT_NAME>
```

5. Create mandatory directories
```bash
python directory_setup.py
```

6. Train the model and find the best threshold:
```bash
python train_model.py
```

7. Run Full Batch Monitoring:
```bash
python full-batch_monitoring.py
```

8. Run Micro Batch Monitoring:
```bash
python micro-batch_monitoring_tool.py
```

9. Compare the results from full-batch and micro-batch
```bash
python compare_results.py
```

Results such as graphs are saved to the <VARIANT_NAME>/output_graphs/full-batch respective <VARIANT_NAME>/output_graphs/full-batch directories.

## Requirements
- Python 3.10
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- evidently
- imbalanced-learn
- 7-zip

  
