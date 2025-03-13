
from sklearn.metrics import (
    auc, classification_report, confusion_matrix, f1_score, 
    precision_score, recall_score, roc_curve
)



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