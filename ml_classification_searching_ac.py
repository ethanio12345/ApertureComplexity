import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import logging
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.base import clone
import pickle, os

class Classifier:
    def __init__(self, X_train, X_test, y_train, y_test, features):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.models = {}
        self.best_model = None

    def train_logistic_regression(self):
        log_reg = LogisticRegression()
        log_reg.fit(self.X_train_scaled, self.y_train)
        self.models['Logistic Regression'] = {
            'model': log_reg,
            'requires_scaling': True,
            'score': log_reg.score(self.X_test_scaled, self.y_test)
        }
        return log_reg

    def train_random_forest(self):
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = {
            'model': rf,
            'requires_scaling': False,
            'score': rf.score(self.X_test, self.y_test)
        }
        return rf

    def train_xgboost(self):
        xgb_model = xgb.XGBClassifier(estrandom_state=42)
        xgb_model.fit(self.X_train, self.y_train)
        self.models['XGBoost'] = {
            'model': xgb_model,
            'requires_scaling': False,
            'score': xgb_model.score(self.X_test, self.y_test)
        }
        return xgb_model
    
    def train_all_models(self):
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Determine best model
        best_score = -1 # default/initial
        best_model_name = None # default/initial
        
        for name, model_info in self.models.items():
            if model_info['score'] > best_score:
                best_score = model_info['score']
                best_model_name = name
        
        self.best_model = {
            'name': best_model_name,
            'model': self.models[best_model_name]['model'],
            'requires_scaling': self.models[best_model_name]['requires_scaling']
        }
        
        return self.models

    def get_feature_importance(self):
        """Get feature importance from tree-based models"""
        feature_importances = {}
        
        for name, model_info in self.models.items():
            if hasattr(model_info['model'], 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': model_info['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                feature_importances[name] = importances
        
        return feature_importances
    
    def optimize_threshold(self, model_name=None):
        """Test different classification thresholds"""
        if model_name is None and self.best_model is not None:
            model_name = self.best_model['name']
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model_info = self.models[model_name]
        X_test_data = self.X_test_scaled if model_info['requires_scaling'] else self.X_test
        
        y_pred_proba = model_info['model'].predict_proba(X_test_data)[:, 0] # optimise for fails
        
        results = []
        thresholds = np.arange(0.1, 0.9, 0.1)
        best_threshold = 0.5 # default/initial
        best_f1 = 0 # default/initial
        
        for threshold in thresholds:
            y_pred_custom = (y_pred_proba <= threshold).astype(int) # predict failures
            recall = recall_score(self.y_test, y_pred_custom, pos_label=0)
            precision = precision_score(self.y_test, y_pred_custom, pos_label=0, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1_score': f1
            })
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        return {
            'threshold_results': results,
            'best_threshold': best_threshold,
            'best_f1': best_f1
        }
    
    def save_models(self, output_dir=None):
        """
        Save the trained models to a directory.

        Parameters:
        output_dir (str): The directory where the models will be saved. Defaults to 'models/' if None.
        """
        if output_dir is None:
            output_dir = 'models/'
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            with open(os.path.join(output_dir, f'{model_name}.pkl'), 'wb') as f:
                pickle.dump(model, f)

class FeatureEvaluator:
    def __init__(self, df, features, target_column, target_definition):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.target_definition = target_definition
        self.scaler = StandardScaler()
        self.y = self.create_target()
        
    def create_target(self):
        """Create binary target variable based on definition"""
        return (self.df[self.target_column] < self.target_definition).astype(int)

    def evaluate_feature(self, feature, model, apply_scaling=False):
        """
        Evaluate a single feature's predictive power using the provided model
        
        Parameters:
        feature (str): The feature to evaluate
        model: Classifier model object with fit, predict, and predict_proba methods
        apply_scaling (bool): Whether to apply scaling to the feature
        
        Returns:
        dict: Dictionary with evaluation metrics
        """
        X_single = self.df[[feature]].astype(float)
        
        X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
            X_single, self.y, test_size=0.2, random_state=42
        )
        
        # Apply scaling if needed
        if apply_scaling:
            X_train_single_processed = self.scaler.fit_transform(X_train_single)
            X_test_single_processed = self.scaler.transform(X_test_single)
        else:
            X_train_single_processed = X_train_single
            X_test_single_processed = X_test_single
        
        # Clone the model to ensure a fresh instance
        model_clone = clone_model_if_possible(model)
        
        # Fit model
        model_clone.fit(X_train_single_processed, y_train_single)
        
        # Predictions for fails
        y_pred_single = model_clone.predict(X_test_single_processed)
        y_pred_proba_single = model_clone.predict_proba(X_test_single_processed)[:, 0]
        accuracy = model_clone.score(X_test_single_processed, y_test_single)
        report = classification_report(y_test_single, y_pred_single, output_dict=True)
        
        # Calculate optimal threshold
        best_threshold = None
        best_f1 = 0
        
        
        for threshold in np.arange(0.1, 0.9, 0.1):
            y_pred_custom = (y_pred_proba_single >= threshold).astype(int)
            recall = recall_score(y_test_single, y_pred_custom, pos_label=0)
            precision = precision_score(y_test_single, y_pred_custom, pos_label=0, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_single, y_pred_proba_single)
        auc_score = auc(fpr, tpr)
        
        # Find optimal cutoff point based on ROC curve
        optimal_idx = np.argmax(tpr - fpr)
        optimal_cutoff = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
        
        return {
            'model_name': type(model).__name__,
            'accuracy': accuracy,
            'recall_class_0': report['0']['recall'],
            'precision_class_0': report['0']['precision'],
            'f1_class_0': report['0']['f1-score'],
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'auc': auc_score,
            'roc_data': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'optimal_cutoff': optimal_cutoff,
            'test_data': {
                'X_test': X_test_single,
                'y_test': y_test_single,
                'y_pred_proba': y_pred_proba_single
            }
        }

    def evaluate_all_features(self, model, apply_scaling=False):
        """
        Evaluate all features individually using the provided model
        
        Parameters:
        model: Classifier model to use for evaluation
        apply_scaling (bool): Whether to apply scaling to the features
        
        Returns:
        dict: Dictionary of features and their evaluation results
        """
        results = {}
        for feature in self.features:
            results[feature] = self.evaluate_feature(feature, model, apply_scaling)
        return results
    
    def evaluate_features_with_multiple_models(self, models, models_require_scaling=None):
        """
        Evaluate all features with multiple models
        
        Parameters:
        models (list): List of classifier models to use
        models_require_scaling (dict, optional): Dictionary mapping model types to boolean scaling requirement
        
        Returns:
        dict: Nested dictionary of model names, features, and evaluation results
        """
        if models_require_scaling is None:
            models_require_scaling = {}
        
        results = {}
        for model in models:
            model_name = type(model).__name__
            # Determine if this model requires scaling based on the provided dict or defaults
            scaling_required = models_require_scaling.get(model_name, False)
            results[model_name] = self.evaluate_all_features(model, scaling_required)
        
        return results
    
    def find_best_feature(self, model=None, metric='auc', apply_scaling=False):
        """
        Find the best performing feature based on a metric
        
        Parameters:
        model: Classifier model to use (defaults to RandomForest if None)
        metric (str): Metric to use for comparison ('auc', 'f1_class_0', etc.)
        apply_scaling (bool): Whether to apply scaling to the features
        
        Returns:
        dict: Dictionary with best feature info
        """
        # Default to RandomForest if no model provided (for backward compatibility)
        if model is None:
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            
        results = self.evaluate_all_features(model, apply_scaling)
        best_feature = None
        best_score = -1
        
        for feature, result in results.items():
            if result[metric] > best_score:
                best_score = result[metric]
                best_feature = feature
        
        return {
            'feature': best_feature,
            'score': best_score,
            'model': type(model).__name__,
            'results': results[best_feature]
        }
    
    def find_best_feature_across_models(self, models, metric='auc', models_require_scaling=None):
        """
        Find the best performing feature across multiple models
        
        Parameters:
        models (list): List of classifier models to evaluate
        metric (str): Metric to use for comparison
        models_require_scaling (dict, optional): Dictionary mapping model types to boolean scaling requirement
        
        Returns:
        dict: Dictionary with best feature and model info
        """
        if models_require_scaling is None:
            models_require_scaling = {}
            
        all_results = self.evaluate_features_with_multiple_models(models, models_require_scaling)
        
        best_feature = None
        best_model_name = None
        best_score = -1
        
        for model_name, feature_results in all_results.items():
            for feature, result in feature_results.items():
                if result[metric] > best_score:
                    best_score = result[metric]
                    best_feature = feature
                    best_model_name = model_name
        
        return {
            'feature': best_feature,
            'model': best_model_name,
            'score': best_score,
            'results': all_results[best_model_name][best_feature]
        }


def clone_model_if_possible(model):
    """
    Attempt to clone a model if sklearn is available, otherwise return the model
    
    Parameters:
    model: The model to clone
    
    Returns:
    A clone of the model if possible, otherwise the original model
    """
    try:
        return clone(model)
    except (ImportError, ValueError):
        # If sklearn is not available or model is not clonable, return the model itself
        # Note: This might cause issues if the model has already been fit
        return model.__class__(**model.get_params())
    

class ResultVisualizer:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        
    def plot_roc_curve(self, y_test, y_pred_proba, model_name, save=True):
        """Plot ROC curve for a model"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        auc_score = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend(loc="lower right")
        
        if save:
            plt.savefig(f'{self.output_dir}/{model_name.replace(" ", "_").lower()}_roc_curve.png')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_roc_curve(self, feature_name, roc_data, save=True):
        """Plot ROC curve for a single feature"""
        fpr, tpr = roc_data['fpr'], roc_data['tpr']
        auc_score = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{feature_name} ROC Curve')
        plt.legend(loc="lower right")
        
        if save:
            plt.savefig(f'{self.output_dir}/{feature_name}_roc_curve.png')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importances(self, feature_importances, model_name, save=True):
        """Plot feature importances"""
        plt.figure()
        plt.barh(feature_importances['Feature'], feature_importances['Importance'])
        plt.xlabel('Importance')
        plt.title(f'{model_name} Feature Importance')
        
        if save:
            plt.savefig(f'{self.output_dir}/{model_name.replace(" ", "_").lower()}_feature_importance.png')
            plt.close()
        else:
            plt.show()
    
    def plot_threshold_optimization(self, threshold_results, metric='f1_score', save=True):
        """Plot metrics vs threshold"""
        thresholds = [result['threshold'] for result in threshold_results]
        metrics = {
            'precision': [result['precision'] for result in threshold_results],
            'recall': [result['recall'] for result in threshold_results],
            'f1_score': [result['f1_score'] for result in threshold_results]
        }
        
        plt.figure()
        for metric_name, metric_values in metrics.items():
            plt.plot(thresholds, metric_values, marker='o', label=metric_name)
        
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Performance Metrics vs. Threshold')
        plt.legend()
        plt.grid(True)
        
        if save:
            plt.savefig(f'{self.output_dir}/threshold_optimization.png')
            plt.close('all')
        else:
            plt.show()


class ResultPrinter:
    def __init__(self, logger=None):
        self.logger = logger

    def print_model_results(self, model_name, y_test, y_pred, score):
        """Print and log model evaluation results"""
        
        lines = [
            f"{model_name} Results:",
            f"Accuracy: {score:.4f}",
            f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}",
            f"Classification Report: \n{classification_report(y_test, y_pred)}"
        ]
        print(f'\n{model_name} Results:')
        print(f'Accuracy: {score:.4f}')
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
        print('Classification Report:')
        print(classification_report(y_test, y_pred))
        
        if self.logger:
            for line in lines:
                self.logger.info(line)


    def print_threshold_results(self, threshold_results, best_threshold, best_f1):
        """Print and log threshold optimization results"""
        self.logger.info("\nThreshold Optimization Results:")
        
        print("\nThreshold Optimization Results:")
        
        for result in threshold_results:
            lines = [
                f"Threshold {result['threshold']:.1f}: Recall {result['recall']:.4f}, ",
                  f"Precision {result['precision']:.4f}, F1 {result['f1_score']:.4f}"
            ]
            if self.logger:
                [self.logger.info(line) for line in lines]
                self.logger.info(f"Best threshold: {best_threshold:.1f} (F1 = {best_f1:.4f})")
        
        print(f"Best threshold: {best_threshold:.1f} (F1 = {best_f1:.4f})")
        
    
    def print_feature_results(self, feature_results):
        """Print, log, and optionally save individual feature results to a file"""
       
        for feature, results in feature_results.items():
            lines = [
                f"\n{feature}:",
                f"  Accuracy: {results['accuracy']:.4f}",
                f"  Recall (class 0): {results['recall_class_0']:.4f}",
                f"  Precision (class 0): {results['precision_class_0']:.4f}",
                f"  F1 (class 0): {results['f1_class_0']:.4f}",
                f"  AUC: {results['auc']:.4f}",
                f"  Best threshold: {results['best_threshold']:.1f}",
                f"  Best F1 with threshold: {results['best_f1']:.4f}",
                f"  Optimal cutoff from ROC: {results['optimal_cutoff']:.4f}"
            ]
            
            if self.logger:
                for line in lines:
                    self.logger.info(line)
    
        # output_text = "\n".join(output_lines)
        # print(output_text)

    
    def print_best_feature(self, best_feature_result, metric='auc'):
        """Print and log best feature results"""
        feature = best_feature_result['feature']
        score = best_feature_result['score']
        cutoff = best_feature_result['results']['optimal_cutoff']
        
        output = [
            f"Best Feature by {metric}: {feature} (Score = {score:.4f})"
            f"For {feature}, use a cutoff of {cutoff:.4f}"
            f"When {feature} >= {cutoff:.4f}, predict Pass Rate > 95"
        ]
        
        print(output)
    
        if self.logger:
            for line in output:
                self.logger.info(line)


def main():
    # Set up logging
    logging.basicConfig(filename='ml_classification.log', level=logging.INFO, filemode='w')
    logger = logging.getLogger()

    # Load data
    df = pd.read_csv('matched_rows_arccheck - cleaned_backup.csv')
    target_col = 'Pass Rate'
    target_threshold = 95
    
    # Create target variable
    df['Target'] = (df[target_col] > target_threshold).astype(int)
    df = df.replace({" nan": np.nan}).dropna()

    # Define features
    features = ['PyComplexityMetric', 'MeanAreaMetricEstimator', 
                'AreaMetricEstimator', 'ApertureIrregularityMetric']
    
    # Prepare data
    X = df[features].dropna().astype(float)
    y = df['Target'].astype(float)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize classes
    classifier = Classifier(X_train, X_test, y_train, y_test, features)
    feature_evaluator = FeatureEvaluator(df, features, target_col, target_threshold)
    result_printer = ResultPrinter(logger)
    visualizer = ResultVisualizer(output_dir='plots')

    # Train all models
    models = classifier.train_all_models()

    # Print model results
    for name, model_info in models.items():
        X_eval = classifier.X_test_scaled if model_info['requires_scaling'] else X_test
        y_pred = model_info['model'].predict(X_eval)
        y_pred_proba = model_info['model'].predict_proba(X_eval)[:, 0]
        
        result_printer.print_model_results(
            name, y_test, y_pred, model_info['score']
        )
        visualizer.plot_roc_curve(y_test, y_pred_proba, name, save=True)
        
    # Get and print feature importances for tree-based models
    feature_importances = classifier.get_feature_importance()
    for model_name, importances in feature_importances.items():
        logger.info(f"\n{model_name} Feature Importances:")
        logger.info(importances)
        visualizer.plot_feature_importances(importances, model_name)
    
    # Optimize threshold for best model
    if classifier.best_model:
        best_model_name = classifier.best_model['name']
        threshold_results = classifier.optimize_threshold(best_model_name)
        result_printer.print_threshold_results(
            threshold_results['threshold_results'],
            threshold_results['best_threshold'],
            threshold_results['best_f1']
        )
        visualizer.plot_threshold_optimization(threshold_results['threshold_results'])
    
        classifier.save_models()

    # SECTION: EVALUATE INDIVIDUAL FEATURES WITH MULTIPLE MODELS
    
    # Define models to test with
    models_to_evaluate = [
        LogisticRegression(max_iter=1000, random_state=42),
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
    for model_name, feature_results in multi_model_results.items():
        print(f"\n--- Model: {model_name} ---")
        for feature, results in feature_results.items():
            lines = [
                    f"  {feature}:",
                    f"    AUC: {results['auc']:.4f}",
                    f"    Recall (class 0): {results['recall_class_0']:.4f}"
                    f"    F1 Score: {results['f1_class_0']:.4f}",
            ]
            # Plot ROC curve for this model/feature combination
            visualizer.plot_feature_roc_curve(
                f"{feature}_{model_name}", 
                results['roc_data'],
                save=True
            )
            
            # Log results
            if logger:
                for line in lines:
                    logger.info(line)
    
    # Find best feature across all models
    best_across_models = feature_evaluator.find_best_feature_across_models(
        models_to_evaluate, 
        metric='recall_class_0',
        models_require_scaling=models_require_scaling
    )
    
    lines = [
        "\n==== Best Feature Across All Models ====",
        f"Feature: {best_across_models['feature']}",
        f"Model: {best_across_models['model']}",
        f"AUC Score: {best_across_models['score']:.4f}",
        f"Optimal Cutoff: {best_across_models['results']['optimal_cutoff']:.4f}"
    ]

    if logger:
        for line in lines:
            logger.info(line)
    
if __name__ == '__main__':
    main()