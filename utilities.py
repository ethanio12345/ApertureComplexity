import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.base import clone

from sklearn.metrics import (
    auc, classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_curve
)
from sklearn.preprocessing import StandardScaler

from dataclasses import dataclass





def optimize_threshold(y_true: np.ndarray, 
                       y_pred_proba: np.ndarray, 
                       positive_class: int = 0, 
                       focus_metric: str = 'recall', 
                       step: float = 0.05) -> Dict:
    """
    Find the optimal classification threshold based on the specified metric.
    
    Args:
        y_true: True class labels
        y_pred_proba: Probability predictions for the positive class
        positive_class: The class index to be treated as the positive class
        focus_metric: 'recall', 'precision', or 'f1' to optimize for
        step: Threshold step size
        
    Returns:
        Dictionary containing threshold evaluation results
    """
    results = []
    thresholds = np.arange(0.1, 0.95, step)
    
    # Define best metric tracker based on focus
    best_value = -1
    best_threshold = 0.5
    
    for threshold in thresholds:
        # Convert probabilities to predictions using threshold
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics for the positive class
        recall = recall_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        precision = precision_score(y_true, y_pred, pos_label=positive_class, zero_division=0)
        
        # Calculate F1 score, handle division by zero
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        result = {
            'threshold': threshold,
            'recall': recall,
            'precision': precision,
            'f1_score': f1
        }
        results.append(result)
        
        # Update best threshold based on focus metric
        if focus_metric == 'recall' and recall > best_value:
            best_value = recall
            best_threshold = threshold
        elif focus_metric == 'precision' and precision > best_value:
            best_value = precision
            best_threshold = threshold
        elif focus_metric == 'f1' and f1 > best_value:
            best_value = f1
            best_threshold = threshold
    
    return {
        'threshold_results': results,
        'best_threshold': best_threshold,
        'best_value': best_value,
        'focus_metric': focus_metric
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
    
