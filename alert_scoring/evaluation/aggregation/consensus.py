"""
Consensus-based aggregation strategies.
"""

from typing import Dict
import pandas as pd
import numpy as np
from loguru import logger

from alert_scoring.evaluation.aggregation.strategies import AggregationStrategy


class MedianConsensus(AggregationStrategy):
    """
    Median-based consensus aggregation.
    
    Uses median to be robust against outliers.
    """
    
    def aggregate(
        self,
        submissions: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """Aggregate using weighted median."""
        
        if not submissions:
            raise ValueError("No submissions provided")
        
        all_alert_ids = set()
        for df in submissions.values():
            all_alert_ids.update(df['alert_id'])
        
        aggregated_scores = []
        
        for alert_id in all_alert_ids:
            scores_with_weights = []
            
            for submitter_id, df in submissions.items():
                alert_scores = df[df['alert_id'] == alert_id]
                
                if len(alert_scores) > 0:
                    score = alert_scores.iloc[0]['score']
                    weight = weights.get(submitter_id, 1.0)
                    scores_with_weights.append((score, weight))
            
            if scores_with_weights:
                scores, ws = zip(*scores_with_weights)
                aggregated_score = self._weighted_median(list(scores), list(ws))
                
                aggregated_scores.append({
                    'alert_id': alert_id,
                    'aggregated_score': aggregated_score,
                    'num_submissions': len(scores_with_weights),
                    'variance': np.var(scores)
                })
        
        return pd.DataFrame(aggregated_scores)
    
    def _weighted_median(self, values: list, weights: list) -> float:
        """Compute weighted median."""
        sorted_pairs = sorted(zip(values, weights))
        values_sorted = [v for v, w in sorted_pairs]
        weights_sorted = [w for v, w in sorted_pairs]
        
        cumsum = np.cumsum(weights_sorted)
        total = cumsum[-1]
        
        median_idx = np.searchsorted(cumsum, total / 2.0)
        return values_sorted[median_idx]
    
    def compute_weights(
        self,
        quality_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute weights from quality scores."""
        weights = {}
        
        for submitter_id, scores in quality_scores.items():
            total_score = sum(scores.values())
            weights[submitter_id] = max(total_score, 0.01)
        
        return weights