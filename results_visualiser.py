import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    auc, classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_curve
)

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