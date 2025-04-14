import os
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def print_feature_info():
    missing = fraud_data_f.isnull().sum()
    print("\nMissing values per column:\n", missing)
    print("\nMissing value percentage per column:\n", (fraud_data_f.isnull().mean() * 100).round(2))

    duplicates = fraud_data_f.duplicated().sum()
    print(f"Number of duplicated rows: {duplicates}")

    print("\nData types:\n", X.dtypes)


fraud_data = pd.read_csv('../data/FiFAR/Base.csv').head(100000)
fraud_data_f = pd.read_csv('../data/FiFAR/Base.csv').tail(900000)
fraud_features = ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count',
       'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w']


tail_y = fraud_data_f['fraud_bool']
tail_X = fraud_data_f[fraud_features]


y = fraud_data['fraud_bool']
X = fraud_data[fraud_features]

print("\nValue counts:\n", y.value_counts())

print_feature_info()

sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X, y)

print("\nOriginal class distribution:\n", y.value_counts())
print("\nResampled class distribution:\n", y_resampled.value_counts())


train_X, test_X, train_y, test_y = train_test_split(X_resampled, y_resampled, random_state=42, test_size=0.2)


model = RandomForestClassifier(random_state=42)
model.fit(train_X, train_y)

prediction = model.predict(tail_X)
print("Accuracy: " + str(accuracy_score(tail_y, prediction)))
print("\nClassification Report:\n", classification_report(tail_y, prediction))


column_mapping = ColumnMapping(
    numerical_features=fraud_features,
    categorical_features=[],
)
report = Report([
    DataDriftPreset(),
])
chunk_size = 200000
for i in range(0, len(fraud_data_f), chunk_size):
    current_chunk = fraud_data_f.iloc[i:i + chunk_size]

    report.run(
        current_data=current_chunk,
        reference_data=fraud_data,
        column_mapping=column_mapping
    )
    result = report.as_dict()

    data_drift = False

    for metric in result["metrics"]:
        if metric["metric"] == "DataDriftTable":
            drift_by_columns = metric["result"].get("drift_by_columns", {})
            for feature_name, feature_data in drift_by_columns.items():
                drift_score = feature_data["drift_score"]
                if drift_score > 0.1:
                    print(f"⚠️ Drift in '{feature_name}': {drift_score:.3f}")
                    data_drift = True

    if not data_drift:
        print("✅ No significant drift detected.")

    reference_data = pd.concat([fraud_data, current_chunk], ignore_index=True)




report.run(current_data=current_chunk,  reference_data=fraud_data, column_mapping=column_mapping)
report.save_html("file.html")
print("HTML-report saved in:", os.path.abspath("file.html"))