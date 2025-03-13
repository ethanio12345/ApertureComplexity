import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass

from utilities import *



@dataclass
class ModelConfig:
    """Configuration for classifier models."""
    name: str
    requires_scaling: bool


@dataclass
class ClassifierConfig:
    """Configuration for the Classifier class."""
    target_column: str
    target_threshold: float
    failure_class: int = 0  # Default: 0 = failure, 1 = success
    output_dir: str = "models"
    random_state: int = 42


class Classifier:
    """
    A class to train and evaluate multiple classification models.
    
    This class handles scaling, model training, threshold optimization,
    and feature importance analysis with a focus on failure detection.
    """
    
    def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series, features: List[str],
                 config: ClassifierConfig = None):
        """
        Initialize the classifier with training and test data.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            features: List of feature names
            config: Configuration parameters
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.features = features
        
        # Use default config if none provided
        if config is None:
            self.config = ClassifierConfig(
                target_column="Target",
                target_threshold=0.5,
                failure_class=0,
                output_dir="models",
                random_state=42
            )
        else:
            self.config = config
            
        # Set up scaling
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Initialize model storage
        self.models = {}
        self.best_model = None
        self.failure_class = self.config.failure_class
        # Create output directory if it doesn't exist
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Logger for this class
        self.logger = logging.getLogger('ml_classification.classifier')

    def train_logistic_regression(self) -> LogisticRegression:
        """
        Train a logistic regression model.
        
        Returns:
            Trained LogisticRegression model
        """
        self.logger.info("Training Logistic Regression model...")
        try:
            # Class weight='balanced' to handle imbalanced data
            log_reg = LogisticRegression(
                class_weight='balanced', 
                max_iter=1000, 
                random_state=self.config.random_state
            )
            log_reg.fit(self.X_train_scaled, self.y_train)
            
            # Score based on recall of failures (class 0)
            y_pred = log_reg.predict(self.X_test_scaled)
            failure_recall = recall_score(
                self.y_test, y_pred, pos_label=self.config.failure_class
            )
            
            self.models['Logistic Regression'] = {
                'model': log_reg,
                'requires_scaling': True,
                'score': failure_recall,
                'scoring_metric': 'failure_recall'
            }
            self.logger.info(f"Logistic Regression trained successfully. Recall: {failure_recall:.4f}")
            return log_reg
            
        except Exception as e:
            self.logger.error(f"Error training Logistic Regression: {str(e)}")
            raise

    def train_random_forest(self) -> RandomForestClassifier:
        """
        Train a random forest classifier.
        
        Returns:
            Trained RandomForestClassifier model
        """
        self.logger.info("Training Random Forest model...")
        try:
            # Increased weight for failures via class_weight
            rf = RandomForestClassifier(
                n_estimators=200, 
                class_weight='balanced', 
                random_state=self.config.random_state
            )
            rf.fit(self.X_train, self.y_train)
            
            # Score based on recall of failures
            y_pred = rf.predict(self.X_test)
            failure_recall = recall_score(
                self.y_test, y_pred, pos_label=self.config.failure_class
            )
            
            self.models['Random Forest'] = {
                'model': rf,
                'requires_scaling': False,
                'score': failure_recall,
                'scoring_metric': 'failure_recall'
            }
            self.logger.info(f"Random Forest trained successfully. Recall: {failure_recall:.4f}")
            return rf
            
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {str(e)}")
            raise

    def train_xgboost(self) -> xgb.XGBClassifier:
        """
        Train an XGBoost classifier.
        
        Returns:
            Trained XGBClassifier model
        """
        self.logger.info("Training XGBoost model...")
        try:
            # Calculate class weight for failures
            weight = np.sum(self.y_train == 1) / np.sum(self.y_train == 0) if np.sum(self.y_train == 0) > 0 else 1.0
            
            xgb_model = xgb.XGBClassifier(
                scale_pos_weight=weight,
                random_state=self.config.random_state
            )
            xgb_model.fit(self.X_train, self.y_train)
            
            # Score based on recall of failures
            y_pred = xgb_model.predict(self.X_test)
            failure_recall = recall_score(
                self.y_test, y_pred, pos_label=self.config.failure_class
            )
            
            self.models['XGBoost'] = {
                'model': xgb_model,
                'requires_scaling': False,
                'score': failure_recall,
                'scoring_metric': 'failure_recall'
            }
            self.logger.info(f"XGBoost trained successfully. Recall: {failure_recall:.4f}")
            return xgb_model
            
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {str(e)}")
            raise
    
    def train_all_models(self) -> Dict:
        """
        Train all classification models.
        
        Returns:
            Dictionary containing all trained models
        """
        self.logger.info("Training all models...")
        try:
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
            
            if best_model_name:
                self.best_model = {
                    'name': best_model_name,
                    'model': self.models[best_model_name]['model'],
                    'requires_scaling': self.models[best_model_name]['requires_scaling']
                }
                self.logger.info(f"Best model: {best_model_name} with score: {best_score:.4f}")
            else:
                self.logger.warning("No best model could be determined")
            
            return self.models
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Get feature importance from tree-based models.
        
        Returns:
            Dictionary with model names as keys and feature importance dataframes as values
        """
        feature_importances = {}
        
        for name, model_info in self.models.items():
            if hasattr(model_info['model'], 'feature_importances_'):
                importances = pd.DataFrame({
                    'Feature': self.features,
                    'Importance': model_info['model'].feature_importances_
                }).sort_values('Importance', ascending=False)
                feature_importances[name] = importances
        
        return feature_importances
    
    def optimize_threshold_for_model(self, model_name: Optional[str] = None, focus_metric: str = 'recall') -> Dict:
        """
        Test different classification thresholds to optimize for failure detection.
        
        Args:
            model_name: Name of model to optimize (uses best model if None)
            focus_metric: 'recall', 'precision', or 'f1'
            
        Returns:
            Dictionary with threshold optimization results
        """
        if model_name is None and self.best_model is not None:
            model_name = self.best_model['name']
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
            
        model_info = self.models[model_name]
        X_test_data = self.X_test_scaled if model_info['requires_scaling'] else self.X_test
        
        # Get probabilities for the failure class
        y_pred_proba = model_info['model'].predict_proba(X_test_data)[:, self.config.failure_class]
        
        # Use the common threshold optimization function
        return optimize_threshold(
            self.y_test, 
            y_pred_proba, 
            positive_class=self.config.failure_class,
            focus_metric=focus_metric
        )
    
    def save_models(self, optimized: bool = True) -> None:
        """
        Save trained models to disk.
        
        Args:
            optimized: Whether to optimize thresholds before saving
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.config.output_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger.info(f"Saving models to {save_dir}...")
        
        try:
            # Save the scaler for future use
            scaler_path = os.path.join(save_dir, "scaler.joblib")
            joblib.dump(self.scaler, scaler_path)
            
            if optimized:
                optimized_thresholds = {}
                for model_name in self.models:
                    optimized_thresholds[model_name] = self.optimize_threshold_for_model(model_name)
                
                # Save the threshold information
                threshold_path = os.path.join(save_dir, "thresholds.json")
                with open(threshold_path, 'w') as f:
                    # Convert numpy values to native Python types for JSON serialization
                    json_safe = {}
                    for name, data in optimized_thresholds.items():
                        json_safe[name] = {
                            "best_threshold": float(data["best_threshold"]),
                            "best_value": float(data["best_value"]),
                            "focus_metric": data["focus_metric"]
                        }
                    json.dump(json_safe, f, indent=2)
                
                # Save models with threshold information
                for model_name, model_info in self.models.items():
                    model = model_info['model']
                    # Add threshold as an attribute to the model
                    model.optimized_threshold = optimized_thresholds[model_name]['best_threshold']
                    model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.joblib")
                    joblib.dump(model, model_path)
                    
                    self.logger.info(f"Saved {model_name} with optimized threshold: {model.optimized_threshold:.4f}")
            else:
                # Save models without threshold optimization
                for model_name, model_info in self.models.items():
                    model = model_info['model']
                    model_path = os.path.join(save_dir, f"{model_name.replace(' ', '_')}.joblib")
                    joblib.dump(model, model_path)
                    
                    self.logger.info(f"Saved {model_name}")
                
            # Save metadata about features and config
            metadata = {
                "features": self.features,
                "failure_class": self.failure_class,
                "requires_scaling": {
                    name: info["requires_scaling"] for name, info in self.models.items()
                },
                "best_model": self.best_model["name"] if self.best_model else None
            }
            
            metadata_path = os.path.join(save_dir, "metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            self.logger.info(f"Saved model metadata to {metadata_path}")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
