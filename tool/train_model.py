import json
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report, \
    confusion_matrix
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score

from CONSTANT_VALUES import *

# Load the data
reference_fraud_data = pd.read_csv(PATH_REFERENCE_DATASET)
new_fraud_data = pd.read_csv(PATH_NEW_DATASET)

# Split the data into features and target variable
target = reference_fraud_data['fraud_bool']
features = reference_fraud_data.drop(columns=['fraud_bool', 'month']).select_dtypes(include=[np.number])

# Remove constant features
constant_features = features.columns[features.nunique() < 50]
features = features.drop(columns=constant_features)

# Feature selection with ANOVA
selector = SelectKBest(score_func=f_classif, k=12)
X_selected = selector.fit_transform(features, target)
selected_columns = selector.get_support(indices=True)
X = features.iloc[:, selected_columns]
y = target

FRAUD_FEATURES = X.columns.tolist()
print("Selected features: ", FRAUD_FEATURES)

# Data balancing
over = SMOTE(sampling_strategy=0.5)
under = RandomUnderSampler(sampling_strategy=0.1)
pipeline = Pipeline([('under', under), ('over', over)])
temp_train_X, temp_train_y = pipeline.fit_resample(X, y)

# Split the data into training and testing sets
train_X, test_X, train_y, test_y = train_test_split(temp_train_X, temp_train_y, train_size=0.8, random_state=42)


# Train a Random Forest Classifier
model = RandomForestClassifier(max_depth=4, random_state=42)
model.fit(train_X, train_y)


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
cv_score = cross_val_score(model, train_X, train_y, cv=cv, scoring='roc_auc').mean()
predictions = model.predict(test_X)
roc_score = roc_auc_score(test_y, predictions)
report = classification_report(test_y, predictions, output_dict=True)

conf_matrix = confusion_matrix(test_y, predictions)
metrics_output = {
    "cross_val_score": cv_score,
    "roc_auc_score": roc_score,
    "classification_report": report,
    "confusion_matrix": conf_matrix.tolist()
}
print(metrics_output)

test_X_prob = test_X.head(TEST_SIZE)
test_y_prob = test_y.head(TEST_SIZE)

probabilities = model.predict_proba(test_X_prob)[:, 1]
thresholds = np.arange(0.000, 1.000, 0.001)

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
print("-" * 40)

best_precision = (0, 0, 0, 0)
best_f1Score = (0, 0, 0, 0)
best_recall = (0, 0, 0, 0)
for threshold in thresholds:
    predictions = (probabilities > threshold).astype(int)
    precision = precision_score(test_y_prob, predictions, zero_division=0)
    recall = recall_score(test_y_prob, predictions, zero_division=0)
    f1 = f1_score(test_y_prob, predictions, zero_division=0)
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
sampled_new_data = new_fraud_data
test_simulation_data_X = pd.concat([test_X.tail(TEST_SIZE), sampled_new_data[FRAUD_FEATURES]], ignore_index=True)
test_simulation_data_y = pd.concat([test_y.tail(TEST_SIZE), sampled_new_data['fraud_bool']], ignore_index=True)

for i in range(0, len(test_simulation_data_X), CHUNK_SIZE):
    print("Processing chunk: ", i // CHUNK_SIZE, "////////////////////////////////////////")
    current_chunk = test_simulation_data_X.iloc[i:i + CHUNK_SIZE]
    current_chunk_target = test_simulation_data_y.iloc[i:i + CHUNK_SIZE]

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
test_simulation_data_X.to_csv(PATH_TEST_X, index=True)
test_simulation_data_y.to_csv(PATH_TEST_Y, index=True)
train_X.to_csv(PATH_TRAIN_X, index=True)

with open(PATH_MODEL_CONFIG, "w") as f:
    json.dump({
        "best_threshold": best_threshold,
        "fraud_features": FRAUD_FEATURES
    }, f)