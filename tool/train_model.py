import json

import numpy as np
import pandas as pd
import sklearn
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from CONSTANT_VALUES import *


# Function to print feature information
def print_feature_info():
    missing = fraud_data.isnull().sum()
    print("\nMissing values per column:\n", missing)
    print("\nMissing value percentage per column:\n", (fraud_data.isnull().mean() * 100).round(2))

    duplicates = fraud_data.duplicated().sum()
    print(f"Number of duplicated rows: {duplicates}")

    print("\nData types:\n", X.dtypes)



# Load the data
fraud_data = pd.read_csv('../data/FiFAR/Base.csv')


# Split the data into features and target variable
y = fraud_data['fraud_bool']
X = fraud_data[FRAUD_FEATURES]

# Split the data into training and testing sets
train_size = int(len(X) * 0.8)

train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
#train_X, test_X = X[:train_size], X[train_size:]
#train_y, test_y = y[:train_size], y[train_size:]

# Resample the training data using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(train_X, train_y)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_resampled, y_resampled)

# Print the target counts
print(train_y.value_counts())

probabilities = model.predict_proba(test_X.head(TEST_SIZE))[:, 1]
thresholds = np.arange(0.000, 1.000, 0.001)

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
print("-" * 40)

best_precision = (0, 0, 0, 0)
best_f1Score = (0, 0, 0, 0)
best_recall = (0, 0, 0, 0)
for threshold in thresholds:
    predictions = (probabilities > threshold).astype(int)
    precision = precision_score(test_y.head(TEST_SIZE), predictions, zero_division=0)
    recall = recall_score(test_y.head(TEST_SIZE), predictions, zero_division=0)
    f1 = f1_score(test_y.head(TEST_SIZE), predictions, zero_division=0)
    if f1 > best_f1Score[3]: best_f1Score = (threshold, precision, recall, f1)
    if precision > best_precision[1]: best_precision = (threshold, precision, recall, f1)
    if recall > best_recall[2]: best_recall = (threshold, precision, recall, f1)
    print(f"{threshold:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

# Print the best thresholds
print("\nBest F1 Score: ", best_f1Score[0], "; With Precision: ", best_f1Score[1], ";  Recall: ", best_f1Score[2],
      ";  F1: ", best_f1Score[3])
print("Best Precision: ", best_precision[0], "; With Precision: ", best_precision[1], ";  Recall: ",
      best_precision[2], ";  F1: ", best_precision[3])
print("Best Recall: ", best_recall[0], "; With Precision: ", best_recall[1], ";  Recall: ", best_recall[2],
      ";  F1: ", best_recall[3])

best_threshold = best_f1Score[0]

precision_list = []
recall_list = []

for i in range(TEST_SIZE, len(test_X), CHUNK_SIZE):
    print("Processing chunk: ", i // CHUNK_SIZE, "////////////////////////////////////////")
    current_chunk = test_X.iloc[i:i + CHUNK_SIZE]
    current_chunk_target = test_y.iloc[i:i + CHUNK_SIZE]

    probabilities = model.predict_proba(current_chunk)[:, 1]
    iteration_predictions = (probabilities > best_threshold).astype(int)

    # Calculate metrics
    precision = precision_score(current_chunk_target, iteration_predictions, zero_division=0)
    recall = recall_score(current_chunk_target, iteration_predictions, zero_division=0)
    precision_list.append(precision)
    recall_list.append(recall)

    print("Precision: ", precision, "Recall: ", recall)

pd.DataFrame({
    "precision": precision_list,
    "recall": recall_list
}).to_csv(PATH_METRICS, index=True)

test_X[TEST_SIZE:].to_csv(PATH_TEST_X, index=True)
test_y[TEST_SIZE:].to_csv(PATH_TEST_Y, index=True)
train_X.to_csv(PATH_TRAIN_X, index=True)

with open(PATH_BEST_THRESHOLD, "w") as f:
    json.dump({"best_threshold": best_threshold}, f)