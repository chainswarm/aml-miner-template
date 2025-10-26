from typing import Dict
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from loguru import logger


class GroundTruthValidator:
    def __init__(self, min_samples: int = 10):
        self.min_samples = min_samples
    
    def validate(self, scores_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> Dict:
        merged = scores_df.merge(
            ground_truth_df[['alert_id', 'is_sar_filed']],
            on='alert_id',
            how='inner'
        )
        
        if len(merged) == 0:
            logger.error("No matching alerts between predictions and ground truth")
            return {
                'score': 0.0,
                'metrics': {},
                'passed': False
            }
        
        if len(merged) < self.min_samples:
            logger.warning(f"Only {len(merged)} samples, minimum {self.min_samples} required")
            return {
                'score': 0.0,
                'metrics': {'n_samples': len(merged)},
                'passed': False
            }
        
        y_true = merged['is_sar_filed'].values
        y_pred = merged['score'].values
        
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            logger.error("Ground truth has only one class")
            return {
                'score': 0.0,
                'metrics': {
                    'n_samples': len(merged),
                    'n_positive': int(y_true.sum())
                },
                'passed': False
            }
        
        auc_roc = roc_auc_score(y_true, y_pred)
        auc_pr = average_precision_score(y_true, y_pred)
        
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1 = np.max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5
        
        score = (0.3 * auc_roc) + (0.2 * auc_pr)
        
        metrics = {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'best_f1': float(best_f1),
            'best_threshold': float(best_threshold),
            'n_samples': int(len(merged)),
            'n_positive': int(y_true.sum()),
            'positive_rate': float(y_true.mean())
        }
        
        logger.info(f"Ground truth validation: AUC-ROC={auc_roc:.3f}, AUC-PR={auc_pr:.3f}, score={score:.3f}")
        
        return {
            'score': score,
            'metrics': metrics,
            'passed': True
        }
    
    def compare_models(self, scores_df_a: pd.DataFrame, scores_df_b: pd.DataFrame, 
                      ground_truth_df: pd.DataFrame) -> Dict:
        result_a = self.validate(scores_df_a, ground_truth_df)
        result_b = self.validate(scores_df_b, ground_truth_df)
        
        if not result_a['passed'] or not result_b['passed']:
            return {
                'comparison_valid': False,
                'reason': 'One or both models failed validation'
            }
        
        metrics_a = result_a['metrics']
        metrics_b = result_b['metrics']
        
        auc_roc_improvement = metrics_b['auc_roc'] - metrics_a['auc_roc']
        auc_pr_improvement = metrics_b['auc_pr'] - metrics_a['auc_pr']
        f1_improvement = metrics_b['best_f1'] - metrics_a['best_f1']
        
        winner = 'model_b' if auc_roc_improvement > 0 else 'model_a'
        
        logger.info(f"Model comparison: AUC-ROC improvement={auc_roc_improvement:.4f}, winner={winner}")
        
        return {
            'comparison_valid': True,
            'winner': winner,
            'metrics_a': metrics_a,
            'metrics_b': metrics_b,
            'improvements': {
                'auc_roc': float(auc_roc_improvement),
                'auc_pr': float(auc_pr_improvement),
                'f1': float(f1_improvement)
            }
        }