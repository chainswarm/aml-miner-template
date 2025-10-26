"""
Weighted ensemble aggregation strategy.
"""

from typing import Dict
import pandas as pd
import numpy as np
from loguru import logger

from alert_scoring.evaluation.aggregation.strategies import AggregationStrategy


class WeightedEnsemble(AggregationStrategy):
    """
    Weighted ensemble aggregation using quality-based weights.
    
    Aggregates scores as: score = Σ(weight_i * score_i) / Σ(weight_i)
    """
    
    def __init__(self, alpha: float = 2.0):
        """
        Args:
            alpha: Exponent for weight amplification (higher = more selective)
        """
        self.alpha = alpha
    
    def aggregate(
        self,
        submissions: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Aggregate submissions using weighted average."""
        
        if not submissions:
            raise ValueError("No submissions provided")
        
        all_alert_ids = set()
        for df in submissions.values():
            all_alert_ids.update(df['alert_id'])
        
        aggregated_scores = []
        
        for alert_id in all_alert_ids:
            weighted_sum = 0.0
            weight_sum = 0.0
            scores_list = []
            
            for submitter_id, df in submissions.items():
                alert_scores = df[df['alert_id'] == alert_id]
                
                if len(alert_scores) > 0:
                    score = alert_scores.iloc[0]['score']
                    weight = weights.get(submitter_id, 0.0)
                    
                    weighted_sum += weight * score
                    weight_sum += weight
                    scores_list.append(score)
            
            if weight_sum > 0:
                aggregated_score = weighted_sum / weight_sum
                variance = np.var(scores_list) if len(scores_list) > 1 else 0.0
                
                aggregated_scores.append({
                    'alert_id': alert_id,
                    'aggregated_score': aggregated_score,
                    'num_submissions': len(scores_list),
                    'variance': variance
                })
        
        return pd.DataFrame(aggregated_scores)
    
    def compute_weights(
        self,
        quality_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute weights from quality scores.
        
        Weight = (total_quality_score) ^ alpha
        """
        weights = {}
        
        for submitter_id, scores in quality_scores.items():
            total_score = sum(scores.values())
            
            weight = total_score ** self.alpha
            weights[submitter_id] = weight
        
        logger.info(f"Computed weights for {len(weights)} submissions")
        
        return weights