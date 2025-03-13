import pickle

# Load the saved models

model_name = "XGBoost"
with open(f'models/{model_name}.pkl', 'rb') as f:
    model = pickle.load(f)

# Define your test case
test_case_pass = {
    'PyComplexityMetric': 0.19990948,
    'MeanAreaMetricEstimator': 69.31572959,
    'AreaMetricEstimator': 419.5091822,
    'ApertureIrregularityMetric': 1.139985207
}


test_case_fail = {
    'PyComplexityMetric': 0.043228985,
    'MeanAreaMetricEstimator': 197.7077636,
    'AreaMetricEstimator': 9445.26388,
    'ApertureIrregularityMetric': 1.233084609
}
# Convert the test case to a Pandas DataFrame
import pandas as pd
test_df = pd.DataFrame([test_case_pass])

# Make predictions using specified model

prediction = model.predict(test_df)
probability = model.predict_proba(test_df)
print(f"Model: {model_name}")
print(f"Predicted outcome: {prediction[0]}")
print(f"Probability of pass: {probability[0, 1]:.4f}")
print(f"Probability of fail: {probability[0, 0]:.4f}")