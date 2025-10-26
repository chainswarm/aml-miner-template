"""Multi-submission aggregation strategies (ensemble, consensus)."""

from alert_scoring.evaluation.aggregation.strategies import AggregationStrategy
from alert_scoring.evaluation.aggregation.ensemble import WeightedEnsemble
from alert_scoring.evaluation.aggregation.consensus import MedianConsensus

__all__ = ['AggregationStrategy', 'WeightedEnsemble', 'MedianConsensus']