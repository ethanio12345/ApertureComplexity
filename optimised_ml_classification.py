import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_score, recall_score, f1_score
import logging

# Set up logging to a file
logging.basicConfig(filename='optimized_classification.log', level=logging.INFO, 
                    filemode='w', format='%(message)s')

# Load and prepare data
df = pd.read_csv('matched_rows.csv')
df['Target'] = (df['Pass Rate'] < 95).astype(int)
df = df.replace({" nan": np.nan}).dropna()

# Define features and target
features = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 
            'AreaMetricEstimator', 'ApertureIrregularityMetric']
best_feature = 'PyComplexityMetric'  # Best single feature based on importance

X = df[features].dropna().astype(float)
y = df['Target'].astype(float)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# Train the best model (Random Forest)
best_model = RandomForestClassifier(n_estimators=100, random_state=42)
best_model.fit(X_train, y_train)

# Get predictions and probability scores
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Find optimal threshold (based on previous results, 0.7 had best recall...means most failures found, but also false alarms)
optimal_threshold = 0.1
y_pred_custom = (y_pred_proba >= optimal_threshold).astype(int)

# Evaluate model with optimal threshold
recall = recall_score(y_test, y_pred_custom)
precision = precision_score(y_test, y_pred_custom)
f1 = f1_score(y_test, y_pred_custom)
accuracy = (y_pred_custom == y_test).mean()

logging.info("BEST MODEL (RANDOM FOREST) WITH OPTIMAL THRESHOLD")
logging.info(f"Optimal threshold: {optimal_threshold:.1f}")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_custom)}")
logging.info(f"Classification Report:\n{classification_report(y_test, y_pred_custom)}")

# Feature importance
importances = pd.DataFrame({
    'Feature': features,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False)
logging.info(f"Feature Importances:\n{importances}")

# BEST INDIVIDUAL FEATURE ANALYSIS
# Using PyComplexityMetric as the best single feature

# Prepare single-feature dataset
X_single = df[[best_feature]].astype(float)

# Split the data
X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
    X_single, y, test_size=0.2, random_state=12
)

# Train Random Forest on single feature
rf_single = RandomForestClassifier(n_estimators=100, random_state=42)
rf_single.fit(X_train_single, y_train_single)

# Get predictions and probability scores
y_pred_proba_single = rf_single.predict_proba(X_test_single)[:, 1]

# Find the optimal cutoff point for the best feature
fpr, tpr, thresholds = roc_curve(y, df[best_feature])
optimal_idx = np.argmax(tpr - fpr)
optimal_cutoff = thresholds[optimal_idx]

# Evaluate single feature with ROC-optimized cutoff
y_pred_single_optimal = (df[best_feature].loc[y_test.index] >= optimal_cutoff).astype(int)
accuracy_single = (y_pred_single_optimal == y_test).mean()
precision_single = precision_score(y_test, y_pred_single_optimal)
recall_single = recall_score(y_test, y_pred_single_optimal)
f1_single = f1_score(y_test, y_pred_single_optimal)

logging.info("\nBEST SINGLE FEATURE ANALYSIS")
logging.info(f"Best feature: {best_feature}")
logging.info(f"Optimal cutoff: {optimal_cutoff:.4f}")
logging.info(f"When {best_feature} >= {optimal_cutoff:.4f}, predict Pass Rate < 95")
logging.info(f"Accuracy: {accuracy_single:.4f}")
logging.info(f"Precision: {precision_single:.4f}")
logging.info(f"Recall: {recall_single:.4f}")
logging.info(f"F1 Score: {f1_single:.4f}")
logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_single_optimal)}")

# Create a simple function for predictions with the best model
def predict_failure(features_df, threshold=optimal_threshold):
    """
    Predict if a sample will have Pass Rate < 95% using the best model.
    
    Parameters:
    features_df (DataFrame): DataFrame with the 4 feature columns
    threshold (float): Probability threshold for classification
    
    Returns:
    numpy.ndarray: Binary predictions (1 = Fail, 0 = Pass)
    """
    # Ensure dataframe has required features
    required_features = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 
                         'AreaMetricEstimator', 'ApertureIrregularityMetric']
    if not all(feature in features_df.columns for feature in required_features):
        raise ValueError(f"Input DataFrame must contain all features: {required_features}")
    
    # Make prediction
    proba = best_model.predict_proba(features_df[required_features])[:, 1]
    return (proba >= threshold).astype(int)

# Create a simple function for predictions with just the best feature
def predict_failure_simple(complexity_value, cutoff=optimal_cutoff):
    """
    Predict if a sample will have Pass Rate < 95% using only PyComplexityMetric.
    
    Parameters:
    complexity_value (float): Value of PyComplexityMetric
    cutoff (float): Cutoff value for classification
    
    Returns:
    int: Binary prediction (1 = Fail, 0 = Pass)
    """
    return 1 if complexity_value >= cutoff else 0

# Example usage
if __name__ == "__main__":
    # Print summary to console
    print(f"Best model (Random Forest) accuracy with threshold {optimal_threshold}: {accuracy:.4f}")
    print(f"Best feature ({best_feature}) prediction with cutoff {optimal_cutoff:.4f}: {accuracy_single:.4f}")
    
    # Example prediction
    print("\nExample prediction:")
    sample = X_test.iloc[0:1]
    prediction = predict_failure(sample)
    print(f"Sample features: {sample.values[0]}")
    print(f"Prediction (1=Fail, 0=Pass): {prediction[0]}")
    
    # Example simple prediction
    complexity_value = sample['PyComplexityMetric'].values[0]
    simple_prediction = predict_failure_simple(complexity_value)
    print(f"Simple prediction using only {best_feature}: {simple_prediction}")