import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix




import logging

# Set up logging to a file
logging.basicConfig(filename='ml_classification.log', level=logging.INFO,filemode='w')


#####################################################

df = pd.read_csv('matched_rows_arccheck.csv')
# df = pd.read_excel('matched_rows_d4.xlsx')


# Assuming your dataframe is called 'df'
# Create the binary target variable
df['Target'] = (df['Pass Rate'] < 95).astype(int)

df = df.replace({" nan": np.nan}).dropna()

# Define features and target
features = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 
            'AreaMetricEstimator', 'ApertureIrregularityMetric']
X = df[features].dropna()
y = df['Target']

# Ensure features and target are floats
X = X.astype(float)
y = y.astype(float)


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#####################################################

# Scale features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#####################################################

# Train models
# 1. Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
log_reg_score = log_reg.score(X_test_scaled, y_test)

# 2. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)

rf.fit(X_train, y_train)  # Note: No scaling needed for tree-based models
rf_score = rf.score(X_test, y_test)

# 3. XGBoost
xgb_model = xgb.XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_score = xgb_model.score(X_test, y_test)

# Evaluate and compare
print(f"Logistic Regression accuracy: {log_reg_score:.4f}")
print(f"Random Forest accuracy: {rf_score:.4f}")
print(f"XGBoost accuracy: {xgb_score:.4f}")

logging.info(f"Logistic Regression accuracy: {log_reg_score:.4f}")
logging.info(f"Random Forest accuracy: {rf_score:.4f}")
logging.info(f"XGBoost accuracy: {xgb_score:.4f}")


# Choose best model and get detailed metrics
best_model = rf  # replace with your best performing model
y_pred = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Feature importance (for Random Forest)
importances = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importances)
logging.info(f"Feature Importances:\n{importances}")


#####################################################

# Adjust classification threshold
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Try different thresholds
from sklearn.metrics import recall_score, precision_score
thresholds = np.arange(0.1, 0.9, 0.1)
for threshold in thresholds:
    y_pred_custom = (y_pred_proba >= threshold).astype(int)
    recall = recall_score(y_test, y_pred_custom)
    precision = precision_score(y_test, y_pred_custom)
    print(f"Threshold {threshold:.1f}: Recall {recall:.4f}, Precision {precision:.4f}")
    logging.info(f"Threshold {threshold:.1f}: Recall {recall:.4f}, Precision {precision:.4f}")




#####################################################

# Test each feature individually
single_feature_results = {}

for feature in features:
    # Prepare single-feature datasets
    X_single = df[[feature]].astype(float)
    
    # Split the data
    X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
        X_single, y, test_size=0.2, random_state=12
    )
    
    # Scale for logistic regression
    X_train_single_scaled = scaler.fit_transform(X_train_single)
    X_test_single_scaled = scaler.transform(X_test_single)
    
    # Train Random Forest on single feature
    rf_single = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    rf_single.fit(X_train_single, y_train_single)
    
    # Evaluate
    y_pred_single = rf_single.predict(X_test_single)
    accuracy = rf_single.score(X_test_single, y_test_single)
    report = classification_report(y_test_single, y_pred_single, output_dict=True)
    
    # Store results
    single_feature_results[feature] = {
        'accuracy': accuracy,
        'recall_class_1': report['1.0']['recall'],
        'precision_class_1': report['1.0']['precision'],
        'f1_class_1': report['1.0']['f1-score']
    }
    
    # Test different thresholds
    y_pred_proba_single = rf_single.predict_proba(X_test_single)[:, 1]
    best_threshold = None
    best_f1 = 0
    
    for threshold in np.arange(0.1, 0.9, 0.1):
        y_pred_custom = (y_pred_proba_single >= threshold).astype(int)
        recall = recall_score(y_test_single, y_pred_custom)
        precision = precision_score(y_test_single, y_pred_custom)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    single_feature_results[feature]['best_threshold'] = best_threshold
    single_feature_results[feature]['best_f1'] = best_f1

# Display results
for feature, results in single_feature_results.items():
    print(f"\n{feature}:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Recall (class 1): {results['recall_class_1']:.4f}")
    print(f"  Precision (class 1): {results['precision_class_1']:.4f}")
    print(f"  F1 (class 1): {results['f1_class_1']:.4f}")
    print(f"  Best threshold: {results['best_threshold']:.1f}")
    print(f"  Best F1 with threshold: {results['best_f1']:.4f}")
    
    logging.info(f"\n{feature}:")
    logging.info(f"  Accuracy: {results['accuracy']:.4f}")
    logging.info(f"  Recall (class 1): {results['recall_class_1']:.4f}")
    logging.info(f"  Precision (class 1): {results['precision_class_1']:.4f}")
    logging.info(f"  F1 (class 1): {results['f1_class_1']:.4f}")
    logging.info(f"  Best threshold: {results['best_threshold']:.1f}")
    logging.info(f"  Best F1 with threshold: {results['best_f1']:.4f}")


#####################################################

# For your best individual feature (e.g., PyComplexityMetric)
best_feature = 'PyComplexityMetric'  # Replace with your actual best feature

# Find the optimal cutoff point
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, df[best_feature])
optimal_idx = np.argmax(tpr - fpr)
optimal_cutoff = thresholds[optimal_idx]

print(f"For {best_feature}, use a cutoff of {optimal_cutoff:.4f}")
logging.info(f"For {best_feature}, use a cutoff of {optimal_cutoff:.4f}")
print(f"When {best_feature} >= {optimal_cutoff:.4f}, predict Pass Rate < 95")
logging.info(f"When {best_feature} >= {optimal_cutoff:.4f}, predict Pass Rate < 95")