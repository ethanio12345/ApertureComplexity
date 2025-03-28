import joblib

# Load the saved models

model_name = "XGBoost"
datestamp = "20250313_165707"
model = joblib.load(f'models/{datestamp}/{model_name}.joblib')

# Define your test case
test_case_pass = {
    'PyComplexityMetric': 0.19990948,
    'MeanAreaMetricEstimator': 69.31572959,
    'AreaMetricEstimator': 419.5091822,
    'ApertureIrregularityMetric': 1.139985207
}


test_case_fail = {
    'PyComplexityMetric': 0.165618745,
    'MeanAreaMetricEstimator': 68.2977662,
    'AreaMetricEstimator': 452.9782629,
    'ApertureIrregularityMetric': 0.906965435
}
# Convert the test case to a Pandas DataFrame
import pandas as pd
test_df = pd.DataFrame([test_case_fail])

# Make predictions using specified model

prediction = model.predict(test_df)
probability = model.predict_proba(test_df)
print(f"Model: {model_name}")
print(f"Predicted outcome: {prediction[0]}")
print(f"Probability of pass: {probability[0, 1]:.4f}")
print(f"Probability of fail: {probability[0, 0]:.4f}")
