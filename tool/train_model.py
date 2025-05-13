import json
import numpy as np
import pandas as pd
import sklearn
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score
from CONSTANT_VALUES import *

# Load data
reference_fraud_data = pd.read_csv(PATH_REFERENCE_DATASET)
new_fraud_data = pd.read_csv(PATH_NEW_DATASET)

# Define features and target
y = reference_fraud_data['fraud_bool']
X = reference_fraud_data[FRAUD_FEATURES]

# Feature selection via ANOVA
selector = SelectKBest(score_func=f_classif, k=20)
X_selected = selector.fit_transform(X, y)
selected_columns = X.columns[selector.get_support()]
X = X[selected_columns]
new_fraud_data = new_fraud_data[selected_columns.tolist() + ['fraud_bool']]

# Split training and testing sets
train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, train_size=0.5, random_state=42)

# Combine undersampling + SMOTE oversampling
under = RandomUnderSampler(sampling_strategy=0.1, random_state=42)
over = SMOTE(sampling_strategy=0.5, random_state=42)
pipeline = Pipeline(steps=[('under', under), ('over', over)])
X_resampled, y_resampled = pipeline.fit_resample(train_X, train_y)

# Train Random Forest
model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    max_depth=4
)
model.fit(X_resampled, y_resampled)

# Threshold tuning
test_X_prob = test_X.head(TEST_SIZE)
test_y_prob = test_y.head(TEST_SIZE)
probabilities = model.predict_proba(test_X_prob)[:, 1]
thresholds = np.arange(0.0, 1.0, 0.001)

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
print("-" * 40)

best_f1Score = (0, 0, 0, 0)
for threshold in thresholds:
    preds = (probabilities > threshold).astype(int)
    precision = precision_score(test_y_prob, preds, zero_division=0)
    recall = recall_score(test_y_prob, preds, zero_division=0)
    f1 = f1_score(test_y_prob, preds, zero_division=0)
    if f1 > best_f1Score[3]:
        best_f1Score = (threshold, precision, recall, f1)
    print(f"{threshold:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

best_threshold = best_f1Score[0]
print(f"\nBest F1 Threshold: {best_threshold:.3f} — Precision: {best_f1Score[1]:.3f}, Recall: {best_f1Score[2]:.3f}, F1: {best_f1Score[3]:.3f}")

# Chunked simulation evaluation
test_simulation_data_X = pd.concat([test_X[TEST_SIZE:TEST_SIZE*2], new_fraud_data[selected_columns]], ignore_index=True)
test_simulation_data_y = pd.concat([test_y[TEST_SIZE:TEST_SIZE*2], new_fraud_data['fraud_bool']], ignore_index=True)

precision_list = []
recall_list = []

for i in range(0, len(test_simulation_data_X), CHUNK_SIZE):
    print(f"Processing chunk {i // CHUNK_SIZE}...")
    chunk_X = test_simulation_data_X.iloc[i:i + CHUNK_SIZE]
    chunk_y = test_simulation_data_y.iloc[i:i + CHUNK_SIZE]
    chunk_probs = model.predict_proba(chunk_X)[:, 1]
    chunk_preds = (chunk_probs > best_threshold).astype(int)

    precision = precision_score(chunk_y, chunk_preds, zero_division=0)
    recall = recall_score(chunk_y, chunk_preds, zero_division=0)

    precision_list.append(precision)
    recall_list.append(recall)

    print(f"Chunk Precision: {precision:.3f}, Recall: {recall:.3f}")

# Save results
pd.DataFrame({
    "precision": precision_list,
    "recall": recall_list
}).to_csv(PATH_METRICS, index=True)
test_simulation_data_X.to_csv(PATH_TEST_X, index=True)
test_simulation_data_y.to_csv(PATH_TEST_Y, index=True)
train_X.to_csv(PATH_TRAIN_X, index=True)

with open(PATH_BEST_THRESHOLD, "w") as f:
    json.dump({"best_threshold": best_threshold}, f)
