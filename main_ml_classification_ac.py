import logging

import numpy as np
import pandas as pd

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
    # Optimize threshold for each model
    for model_name, model_info in models.items():
        threshold_results = classifier.optimize_threshold_for_model(model_name, focus_metric='recall')
        models[model_name]['best_threshold'] = threshold_results['best_threshold']

    # Print model training results
    logger.info("\n==== MODEL TRAINING RESULTS ====")
    
    # Print model results
    for name, model_info in models.items():
        X_eval = classifier.X_test_scaled if model_info['requires_scaling'] else classifier.X_test
        y_pred = model_info['model'].predict(X_eval)
        y_pred_proba = model_info['model'].predict_proba(X_eval)[:, classifier.failure_class]
        
        # Use the optimized threshold for prediction
        y_pred_optimized = (y_pred_proba >= model_info['best_threshold']).astype(int)
    
        result_printer.print_model_results(
            name, classifier.y_test, y_pred_optimized, model_info['score'], classifier.failure_class
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
    for name, model_info in models.items():
        # Specify focus metric explicitly for consistency
        threshold_results = classifier.optimize_threshold_for_model(name, focus_metric='recall')
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
            title_suffix=name
        )
    
        # Save all models
        classifier.save_models()


    print("\nAnalysis complete. Results saved to log file and plots directory.")
    logger.info("\nAnalysis complete.")
    
if __name__ == '__main__':
    main()