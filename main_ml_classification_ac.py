import logging

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split

from classifier_arccheck import Classifier
from feature_evaluators_arccheck import FeatureEvaluator
from results_visualiser import ResultVisualizer
from results_printer import ResultPrinter



def main():
    # Set up logging
    logging.basicConfig(filename='ml_classification.log', level=logging.INFO, filemode='w', format='%(message)s')
    logger = logging.getLogger()

    # Load data
    df = pd.read_csv('matched_rows_arccheck - cleaned_backup.csv')
    target_col = 'Pass Rate'
    target_threshold = 95
    
    # Create target variable
    df['Target'] = (df[target_col] >= target_threshold).astype(int)
    df = df.replace({" nan": np.nan}).dropna()

    # Define features
    features = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 
                'AreaMetricEstimator', 'ApertureIrregularityMetric']
    
    # Prepare data
    X = df[features].dropna().astype(float)
    y = df['Target'].loc[X.index].astype(int)  # Ensure y corresponds to X after dropna

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialize classes
    classifier = Classifier(X_train, X_test, y_train, y_test, features)
    feature_evaluator = FeatureEvaluator(df, features, target_col, target_threshold)
    result_printer = ResultPrinter(logger)
    visualizer = ResultVisualizer(output_dir='plots')

    # Train all models
    models = classifier.train_all_models()
    
    logger.info("\n==== MODEL TRAINING RESULTS ====")
    
    # Print model results
    for name, model_info in models.items():
        X_eval = classifier.X_test_scaled if model_info['requires_scaling'] else classifier.X_test
        y_pred = model_info['model'].predict(X_eval)
        y_pred_proba = model_info['model'].predict_proba(X_eval)[:, classifier.failure_class]
        
        result_printer.print_model_results(
            name, classifier.y_test, y_pred, model_info['score'], classifier.failure_class
        )
        visualizer.plot_roc_curve(classifier.y_test, y_pred_proba, name, failure_class=classifier.failure_class, save=True)
        
    # Get and print feature importances for tree-based models
    feature_importances = classifier.get_feature_importance()
    for model_name, importances in feature_importances.items():
        logger.info(f"\n{model_name} Feature Importances:")
        logger.info(importances)
        visualizer.plot_feature_importances(importances, model_name)
    
    # Optimize threshold for best model
    logger.info("\n==== THRESHOLD OPTIMIZATION ====")
    if classifier.best_model:
        best_model_name = classifier.best_model['name']
        # Specify focus metric explicitly for consistency
        threshold_results = classifier.optimize_threshold_for_model(best_model_name, focus_metric='recall')
        result_printer.print_threshold_results(
            threshold_results['threshold_results'],
            threshold_results['best_threshold'],
            threshold_results['best_value'],
            focus_metric=threshold_results['focus_metric']
        )
        visualizer.plot_threshold_optimization(
            threshold_results['threshold_results'], 
            best_threshold=threshold_results['best_threshold'],
            focus_metric=threshold_results['focus_metric'],
            title_suffix=best_model_name
        )
    
        # Save all models
        classifier.save_models()

    # SECTION: EVALUATE INDIVIDUAL FEATURES WITH MULTIPLE MODELS
    logger.info("\n==== INDIVIDUAL FEATURE EVALUATION ====")
    
    # Define models to test with
    models_to_evaluate = [
        LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        xgb.XGBClassifier(random_state=42)
    ]
    
    # Define which models require scaling
    models_require_scaling = {
        'LogisticRegression': True,
        'RandomForestClassifier': False,
        'XGBClassifier': False
    }
    
    # Evaluate features with multiple models
    multi_model_results = feature_evaluator.evaluate_features_with_multiple_models(
        models_to_evaluate, models_require_scaling
    )
    
    # Print results for each model and feature
    print("\n==== Feature Evaluation Across Multiple Models ====")
    logger.info("\n==== FEATURE EVALUATION ACROSS MULTIPLE MODELS ====")
    for model_name, feature_results in multi_model_results.items():
        print(f"\n--- Model: {model_name} ---")
        logger.info(f"\n--- Model: {model_name} ---")
        result_printer.print_feature_results(feature_results, failure_class=feature_evaluator.failure_class)
        
        for feature, results in feature_results.items():
            # Plot ROC curve for this model/feature combination
            visualizer.plot_feature_roc_curve(
                feature, 
                results['roc_data'],
                model_name=model_name,
                save=True
            )
            
            # Plot threshold analysis for each feature
            visualizer.plot_feature_threshold_analysis(
                feature,
                results,
                save=True
            )
    
    # Find best feature across all models
    best_across_models = feature_evaluator.find_best_feature_across_models(
        models_to_evaluate, 
        metric='recall_class_0',
        models_require_scaling=models_require_scaling
    )
    
    # Print best feature results
    result_printer.print_best_feature(best_across_models, metric='recall_class_0')
    
    # Highlight the best feature from all models with visualization
    best_feature = best_across_models['feature']
    best_model = best_across_models['model']
    best_results = best_across_models['results']
    
    visualizer.plot_feature_roc_curve(
        best_feature,
        best_results['roc_data'],
        model_name=f"BEST-{best_model}",
        save=True
    )
    
    visualizer.plot_feature_threshold_analysis(
        f"BEST-{best_feature}",
        best_results,
        save=True
    )
    
    print("\nAnalysis complete. Results saved to log file and plots directory.")
    logger.info("\nAnalysis complete.")
    
if __name__ == '__main__':
    main()