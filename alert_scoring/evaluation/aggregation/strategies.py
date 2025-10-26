"""
Base strategy interface for multi-submission aggregation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
import pandas as pd


class AggregationStrategy(ABC):
    """Base class for aggregation strategies."""
    
    @abstractmethod
    def aggregate(
        self, 
        submissions: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Aggregate multiple submissions into consensus scores.
        
        Args:
            submissions: {submitter_id: scores_dataframe}
            weights: {submitter_id: weight}
            
        Returns:
            DataFrame with aggregated scores
        """
        pass
    
    @abstractmethod
    def compute_weights(
        self,
        quality_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute submission weights from quality scores.
        
        Args:
            quality_scores: {submitter_id: {metric: score}}
            
        Returns:
            {submitter_id: weight}
        """
        pass