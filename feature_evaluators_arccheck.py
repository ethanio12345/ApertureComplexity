
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    auc, classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utilities import *


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
        
        if apply_scaling:
            X_train_single_processed = self.scaler.fit_transform(X_train_single)
            X_test_single_processed = self.scaler.transform(X_test_single)
        else:
            X_train_single_processed = X_train_single
            X_test_single_processed = X_test_single
        
        model_clone = clone_model_if_possible(model)
        model_clone.fit(X_train_single_processed, y_train_single)
        
        y_pred_proba_single = model_clone.predict_proba(X_test_single_processed)[:, self.failure_class]
        accuracy = model_clone.score(X_test_single_processed, y_test_single)
        report = classification_report(y_test_single, (y_pred_proba_single >= 0.5).astype(int), output_dict=True)
        
        threshold_optimization_results = optimize_threshold(
            y_test_single, 
            y_pred_proba_single, 
            positive_class=self.failure_class,
            focus_metric=focus_metric,
            step=0.05
        )
        
        fpr, tpr, thresholds = roc_curve(y_test_single, y_pred_proba_single, pos_label=self.failure_class)
        auc_score = auc(fpr, tpr)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_cutoff = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
        
        return {
            'model_name': type(model).__name__,
            'accuracy': accuracy,
            'recall_class_0': report[str(self.failure_class)]['recall'],
            'precision_class_0': report[str(self.failure_class)]['precision'],
            'f1_class_0': report[str(self.failure_class)]['f1-score'],
            'best_threshold': threshold_optimization_results['best_threshold'],
            'best_f1': threshold_optimization_results['best_value'],
            'best_recall_threshold': max(threshold_optimization_results['threshold_results'], key=lambda x: x['recall'])['threshold'],
            'max_recall': max(threshold_optimization_results['threshold_results'], key=lambda x: x['recall'])['recall'],
            'auc': auc_score,
            'roc_data': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'optimal_cutoff': optimal_cutoff,
            'threshold_results': threshold_optimization_results['threshold_results'],
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

