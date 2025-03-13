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
import warnings
warnings.filterwarnings('ignore')

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
        # Define the positive class for failure prediction
        self.failure_class = 0  # Assuming 0 = failure, 1 = success

    def train_logistic_regression(self):
        # Class weight='balanced' to handle imbalanced data
        log_reg = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        log_reg.fit(self.X_train_scaled, self.y_train)
        
        # Score based on recall of failures (class 0)
        y_pred = log_reg.predict(self.X_test_scaled)
        failure_recall = recall_score(self.y_test, y_pred, pos_label=self.failure_class)
        
        self.models['Logistic Regression'] = {
            'model': log_reg,
            'requires_scaling': True,
            'score': failure_recall,
            'scoring_metric': 'failure_recall'
        }
        return log_reg

    def train_random_forest(self):
        # Increased weight for failures via class_weight
        rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
        rf.fit(self.X_train, self.y_train)
        
        # Score based on recall of failures
        y_pred = rf.predict(self.X_test)
        failure_recall = recall_score(self.y_test, y_pred, pos_label=self.failure_class)
        
        self.models['Random Forest'] = {
            'model': rf,
            'requires_scaling': False,
            'score': failure_recall,
            'scoring_metric': 'failure_recall'
        }
        return rf

    def train_xgboost(self):
        # Set scale_pos_weight to handle class imbalance
        # Calculate class weight for failures
        weight = np.sum(self.y_train == 1) / np.sum(self.y_train == 0)
        
        xgb_model = xgb.XGBClassifier(
            scale_pos_weight=weight,
            random_state=42
        )
        xgb_model.fit(self.X_train, self.y_train)
        
        # Score based on recall of failures
        y_pred = xgb_model.predict(self.X_test)
        failure_recall = recall_score(self.y_test, y_pred, pos_label=self.failure_class)
        
        self.models['XGBoost'] = {
            'model': xgb_model,
            'requires_scaling': False,
            'score': failure_recall,
            'scoring_metric': 'failure_recall'
        }
        return xgb_model
    
    def train_all_models(self):
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_xgboost()
        
        # Determine best model based on failure recall
        best_score = -1
        best_model_name = None
        
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
    
    def optimize_threshold(self, model_name=None, focus_metric='recall'):
        """
        Test different classification thresholds optimizing for failure detection
        
        Parameters:
        model_name (str): Name of model to optimize
        focus_metric (str): 'recall', 'precision', or 'f1'
        
        Returns:
        dict: Threshold optimization results
        """
        if model_name is None and self.best_model is not None:
            model_name = self.best_model['name']
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model_info = self.models[model_name]
        X_test_data = self.X_test_scaled if model_info['requires_scaling'] else self.X_test
        
        # Get probabilities for class 0 (failures)
        y_pred_proba = model_info['model'].predict_proba(X_test_data)[:, self.failure_class]
        
        results = []
        thresholds = np.arange(0.1, 0.95, 0.05)  # More granular thresholds
        best_threshold = 0.5
        
        # Define metrics to track based on focus
        if focus_metric == 'recall':
            best_metric_val = 0
        elif focus_metric == 'precision':
            best_metric_val = 0
        else:  # f1
            best_metric_val = 0
        
        for threshold in thresholds:
            # Higher threshold means more samples classified as failures
            y_pred_custom = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics for failures
            recall = recall_score(self.y_test, y_pred_custom, pos_label=self.failure_class)
            precision = precision_score(self.y_test, y_pred_custom, pos_label=self.failure_class, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1_score': f1
            })
            
            # Update best threshold based on focus metric
            current_metric = None
            if focus_metric == 'recall':
                current_metric = recall
            elif focus_metric == 'precision':
                current_metric = precision
            else:  # f1
                current_metric = f1
                
            if current_metric > best_metric_val:
                best_metric_val = current_metric
                best_threshold = threshold
        
        return {
            'threshold_results': results,
            'best_threshold': best_threshold,
            'best_value': best_metric_val,
            'focus_metric': focus_metric
        }
    
    def save_models(self, output_dir="models", optimized=True):
        if optimized:
            optimized_thresholds = {}
            for model_name in self.models:
                optimized_thresholds[model_name] = self.optimize_threshold(model_name)
            for model_name, model_info in self.models.items():
                model = model_info['model']
                optimized_threshold = optimized_thresholds[model_name]['best_threshold']
                # Use the optimized threshold to create a new model
                model = clone(model)
                model.threshold = optimized_threshold
                with open(os.path.join(output_dir, f"{model_name}.pkl"), 'wb') as f:
                    pickle.dump(model, f)
        else:
            for model_name, model_info in self.models.items():
                model = model_info['model']
                with open(os.path.join(output_dir, f"{model_name}.pkl"), 'wb') as f:
                    pickle.dump(model, f)

class FeatureEvaluator:
    def __init__(self, df, features, target_column, target_definition):
        self.df = df
        self.features = features
        self.target_column = target_column
        self.target_definition = target_definition
        self.scaler = StandardScaler()
        self.y = self.create_target()
        # Define the failure class
        self.failure_class = 0  # 0 = failure (target < threshold)
        
    def create_target(self):
        """Create binary target variable based on definition: 0=failure, 1=success"""
        return (self.df[self.target_column] >= self.target_definition).astype(int)

    def evaluate_feature(self, feature, model, apply_scaling=False, focus_metric='recall_class_0'):
        """
        Evaluate a single feature's predictive power for failure detection
        
        Parameters:
        feature (str): The feature to evaluate
        model: Classifier model object
        apply_scaling (bool): Whether to apply scaling
        focus_metric (str): Metric to optimize for
        
        Returns:
        dict: Dictionary with evaluation metrics
        """
        X_single = self.df[[feature]].astype(float)
        
        X_train_single, X_test_single, y_train_single, y_test_single = train_test_split(
            X_single, self.y, test_size=0.2, random_state=42, stratify=self.y
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
        
        # Predictions
        y_pred_single = model_clone.predict(X_test_single_processed)
        y_pred_proba_single = model_clone.predict_proba(X_test_single_processed)[:, self.failure_class]
        accuracy = model_clone.score(X_test_single_processed, y_test_single)
        report = classification_report(y_test_single, y_pred_single, output_dict=True)
        
        # Calculate optimal threshold for failure detection
        best_threshold = None
        best_recall = 0
        best_f1 = 0
        best_precision = 0
        thresholds_to_test = np.arange(0.1, 0.95, 0.05)
        
        # Store results for each threshold
        threshold_results = []
        
        for threshold in thresholds_to_test:
            y_pred_custom = (y_pred_proba_single >= threshold).astype(int)
            recall = recall_score(y_test_single, y_pred_custom, pos_label=self.failure_class)
            precision = precision_score(y_test_single, y_pred_custom, pos_label=self.failure_class, zero_division=0)
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            threshold_results.append({
                'threshold': threshold,
                'recall': recall,
                'precision': precision,
                'f1_score': f1
            })
            
            # Update best values
            if recall > best_recall:
                best_recall = recall
            
            if precision > best_precision:
                best_precision = precision
                
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        # Calculate ROC curve focused on failures
        fpr, tpr, thresholds = roc_curve(y_test_single, y_pred_proba_single, pos_label=self.failure_class)
        auc_score = auc(fpr, tpr)
        
        # Find optimal cutoff point based on ROC curve
        optimal_idx = np.argmax(tpr - fpr)
        optimal_cutoff = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
        
        # If focus is on recall, find threshold with best recall
        threshold_for_recall = max(threshold_results, key=lambda x: x['recall'])['threshold']
        max_recall = max(threshold_results, key=lambda x: x['recall'])['recall']
        
        return {
            'model_name': type(model).__name__,
            'accuracy': accuracy,
            'recall_class_0': report[str(self.failure_class)]['recall'],
            'precision_class_0': report[str(self.failure_class)]['precision'],
            'f1_class_0': report[str(self.failure_class)]['f1-score'],
            'best_threshold': best_threshold,
            'best_f1': best_f1,
            'best_recall_threshold': threshold_for_recall,
            'max_recall': max_recall,
            'auc': auc_score,
            'roc_data': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'optimal_cutoff': optimal_cutoff,
            'threshold_results': threshold_results,
            'test_data': {
                'X_test': X_test_single,
                'y_test': y_test_single,
                'y_pred_proba': y_pred_proba_single
            }
        }

    def evaluate_all_features(self, model, apply_scaling=False, focus_metric='recall_class_0'):
        """Evaluate all features individually using the provided model"""
        results = {}
        for feature in self.features:
            results[feature] = self.evaluate_feature(feature, model, apply_scaling, focus_metric)
        return results
    
    def evaluate_features_with_multiple_models(self, models, models_require_scaling=None, focus_metric='recall_class_0'):
        """Evaluate all features with multiple models"""
        if models_require_scaling is None:
            models_require_scaling = {}
        
        results = {}
        for model in models:
            model_name = type(model).__name__
            # Determine if this model requires scaling
            scaling_required = models_require_scaling.get(model_name, False)
            results[model_name] = self.evaluate_all_features(model, scaling_required, focus_metric)
        
        return results
    
    def find_best_feature(self, model=None, metric='recall_class_0', apply_scaling=False):
        """Find the best performing feature for failure detection"""
        # Default to RandomForest if no model provided
        if model is None:
            model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
            
        results = self.evaluate_all_features(model, apply_scaling, metric)
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
    
    def find_best_feature_across_models(self, models, metric='recall_class_0', models_require_scaling=None):
        """Find the best performing feature for failure detection across multiple models"""
        if models_require_scaling is None:
            models_require_scaling = {}
            
        all_results = self.evaluate_features_with_multiple_models(models, models_require_scaling, metric)
        
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
    """Clone a model if possible, otherwise create a new instance"""
    try:
        return clone(model)
    except (ImportError, ValueError):
        try:
            return model.__class__(**model.get_params())
        except:
            # Last resort fallback
            if hasattr(model, '__class__') and hasattr(model, 'get_params'):
                params = {k: v for k, v in model.get_params().items() 
                         if not (isinstance(v, type) or callable(v))}
                return model.__class__(**params)
            else:
                return model
    

class ResultVisualizer:
    def __init__(self, output_dir='plots'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use('ggplot')
        
    def plot_roc_curve(self, y_test, y_pred_proba, model_name, failure_class=0, save=True):
        """Plot ROC curve for a model focused on failure detection"""
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=failure_class)
        auc_score = auc(fpr, tpr)

        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'{model_name} (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} ROC Curve - Failure Detection', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            plt.savefig(f'{self.output_dir}/{model_name.replace(" ", "_").lower()}_roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_roc_curve(self, feature_name, roc_data, model_name="", save=True):
        """Plot ROC curve for a single feature focused on failure detection"""
        fpr, tpr = roc_data['fpr'], roc_data['tpr']
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        title = f'{feature_name} ROC Curve - Failure Detection'
        if model_name:
            title += f' ({model_name})'
        plt.title(title, fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            filename = f'{feature_name.replace(" ", "_").lower()}'
            if model_name:
                filename += f'_{model_name.replace(" ", "_").lower()}'
            plt.savefig(f'{self.output_dir}/{filename}_roc_curve.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_importances(self, feature_importances, model_name, save=True):
        """Plot feature importances for failure prediction"""
        plt.figure(figsize=(10, 6))
        # Sort by importance for better visualization
        feature_importances = feature_importances.sort_values('Importance', ascending=True)
        plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='#1f77b4')
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'{model_name} Feature Importance for Failure Detection', fontsize=14)
        plt.grid(True, axis='x', linestyle='--', alpha=0.7)
        
        # Add importance values
        for i, v in enumerate(feature_importances['Importance']):
            plt.text(v + 0.01, i, f'{v:.4f}', va='center')
        
        if save:
            plt.savefig(f'{self.output_dir}/{model_name.replace(" ", "_").lower()}_feature_importance.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_threshold_optimization(self, threshold_results, best_threshold=None, focus_metric='recall', save=True, title_suffix=""):
        """Plot metrics vs threshold with focus on failure detection"""
        thresholds = [result['threshold'] for result in threshold_results]
        metrics = {
            'precision': [result['precision'] for result in threshold_results],
            'recall': [result['recall'] for result in threshold_results],
            'f1_score': [result['f1_score'] for result in threshold_results]
        }
        
        plt.figure(figsize=(10, 6))
        colors = {'precision': '#1f77b4', 'recall': '#ff7f0e', 'f1_score': '#2ca02c'}
        
        for metric_name, metric_values in metrics.items():
            plt.plot(thresholds, metric_values, marker='o', label=metric_name, color=colors[metric_name])
        
        # Highlight best threshold if provided
        if best_threshold is not None:
            plt.axvline(x=best_threshold, color='r', linestyle='--', 
                       label=f'Best threshold: {best_threshold:.2f}', alpha=0.7)
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        title = 'Failure Detection - Performance Metrics vs. Threshold'
        if title_suffix:
            title += f' - {title_suffix}'
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save:
            filename = 'threshold_optimization'
            if title_suffix:
                filename += f'_{title_suffix.replace(" ", "_").lower()}'
            plt.savefig(f'{self.output_dir}/{filename}.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_feature_threshold_analysis(self, feature_name, feature_results, save=True):
        """Plot detailed threshold analysis for a feature"""
        if 'threshold_results' not in feature_results:
            return
            
        thresholds = [result['threshold'] for result in feature_results['threshold_results']]
        metrics = {
            'precision': [result['precision'] for result in feature_results['threshold_results']],
            'recall': [result['recall'] for result in feature_results['threshold_results']],
            'f1_score': [result['f1_score'] for result in feature_results['threshold_results']]
        }
        
        plt.figure(figsize=(10, 6))
        colors = {'precision': '#1f77b4', 'recall': '#ff7f0e', 'f1_score': '#2ca02c'}
        
        for metric_name, metric_values in metrics.items():
            plt.plot(thresholds, metric_values, marker='o', label=metric_name, color=colors[metric_name])
        
        # Highlight optimal cutoff
        plt.axvline(x=feature_results['optimal_cutoff'], color='r', linestyle='--', 
                   label=f'Optimal cutoff: {feature_results["optimal_cutoff"]:.2f}', alpha=0.7)
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title(f'{feature_name} - Failure Detection Threshold Analysis', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save:
            plt.savefig(f'{self.output_dir}/{feature_name.replace(" ", "_").lower()}_threshold_analysis.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


class ResultPrinter:
    def __init__(self, logger=None):
        self.logger = logger

    def print_model_results(self, model_name, y_test, y_pred, score, failure_class=0):
        """Print and log model evaluation results with focus on failure detection"""
        
        # Get metrics specifically for failure class
        recall = recall_score(y_test, y_pred, pos_label=failure_class)
        precision = precision_score(y_test, y_pred, pos_label=failure_class)
        f1 = f1_score(y_test, y_pred, pos_label=failure_class)
        
        lines = [
            f"\n{model_name} Results (Failure Detection Focus):",
            f"Accuracy: {score:.4f}",
            f"Recall (Failure Class): {recall:.4f}",
            f"Precision (Failure Class): {precision:.4f}",
            f"F1 Score (Failure Class): {f1:.4f}",
            f"Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}",
            f"Classification Report: \n{classification_report(y_test, y_pred)}"
        ]
        
        for line in lines:
            print(line)
        
        if self.logger:
            for line in lines:
                self.logger.info(line)


    def print_threshold_results(self, threshold_results, best_threshold, best_value, focus_metric='recall'):
        """Print and log threshold optimization results for failure detection"""
        print(f"\nThreshold Optimization Results (Focus: {focus_metric}):")
        
        for result in threshold_results:
            line = (f"Threshold {result['threshold']:.2f}: "
                    f"Recall {result['recall']:.4f}, "
                    f"Precision {result['precision']:.4f}, "
                    f"F1 {result['f1_score']:.4f}")
            print(line)
            
            if self.logger:
                self.logger.info(line)
        
        best_line = f"Best threshold for {focus_metric}: {best_threshold:.2f} ({focus_metric} = {best_value:.4f})"
        print(best_line)
        
        if self.logger:
            self.logger.info(best_line)
    
    def print_feature_results(self, feature_results, failure_class=0):
        """Print and log individual feature results with focus on failure detection"""
        print("\n=== Feature Evaluation Results (Failure Detection Focus) ===")
        
        for feature, results in feature_results.items():
            lines = [
                f"\n{feature}:",
                f"  Accuracy: {results['accuracy']:.4f}",
                f"  Recall (Failure Class): {results['recall_class_0']:.4f}",
                f"  Precision (Failure Class): {results['precision_class_0']:.4f}",
                f"  F1 (Failure Class): {results['f1_class_0']:.4f}",
                f"  AUC: {results['auc']:.4f}",
                f"  Best threshold (F1): {results['best_threshold']:.2f} (F1 = {results['best_f1']:.4f})",
                f"  Best threshold (Recall): {results['best_recall_threshold']:.2f} (Recall = {results['max_recall']:.4f})",
                f"  Optimal cutoff from ROC: {results['optimal_cutoff']:.4f}"
            ]
            
            for line in lines:
                print(line)
                
                if self.logger:
                    self.logger.info(line)
    
    def print_best_feature(self, best_feature_result, metric='recall_class_0'):
        """Print and log best feature results for failure detection"""
        feature = best_feature_result['feature']
        score = best_feature_result['score']
        model = best_feature_result['model']
        cutoff = best_feature_result['results']['optimal_cutoff']
        

        
        lines = [
            f"\n=== Best Feature for Failure Detection ({metric}) ===",
            f"Feature: {feature}",
            f"Model: {model}",
            f"Score: {score:.4f}",
            f"For {feature}, optimal cutoff: {cutoff:.4f}",
            f"Rule: When {feature} >= {cutoff:.4f}, likely to predict failure (Pass Rate < 95)"
        ]
        
        for line in lines:
            print(line)
            
            if self.logger:
                self.logger.info(line)


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
        threshold_results = classifier.optimize_threshold(best_model_name, focus_metric='recall')
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