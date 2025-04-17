# Predict on the test set
from sklearn.metrics import precision_score, recall_score, f1_score

probabilities = model.predict_proba(test_X)[:, 1]
thresholds = np.arange(0.000, 1.000, 0.001)

print(f"{'Threshold':<10} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
print("-" * 40)

best_percision = (0, 0, 0, 0)
best_f1Score = (0, 0, 0, 0)
best_recall = (0, 0, 0, 0)
for threshold in thresholds:
    preds = (probabilities > threshold).astype(int)
    precision = precision_score(test_y, preds, zero_division=0)
    recall = recall_score(test_y, preds, zero_division=0)
    f1 = f1_score(test_y, preds, zero_division=0)
    if f1 > best_f1Score[3]: best_f1Score = (threshold, precision, recall, f1)
    if precision > best_percision[1]: best_percision = (threshold, precision, recall, f1)
    if recall > best_recall[2]: best_recall = (threshold, precision, recall, f1)
    print(f"{threshold:<10.3f} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")

# Print the best thresholds
print("\nBest F1 Score: ", best_f1Score[0], "; With Precision: ", best_f1Score[1], ";  Recall: ", best_f1Score[2], ";  F1: ", best_f1Score[3])
print("Best Precision: ", best_percision[0], "; With Precision: ", best_percision[1], ";  Recall: ", best_percision[2], ";  F1: ", best_percision[3])
print("Best Recall: ", best_recall[0], "; With Precision: ", best_recall[1], ";  Recall: ", best_recall[2], ";  F1: ", best_recall[3])
