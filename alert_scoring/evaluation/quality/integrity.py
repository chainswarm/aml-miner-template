from typing import Dict
import pandas as pd
from loguru import logger


class IntegrityValidator:
    def __init__(self, max_latency_ms: int = 100):
        self.max_latency_ms = max_latency_ms
    
    def validate(self, scores_df: pd.DataFrame, alerts_df: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'score': 0.0,
            'checks': {}
        }
        
        if len(scores_df) != len(alerts_df):
            logger.error(f"Completeness check failed: {len(scores_df)} scores vs {len(alerts_df)} alerts")
            results['checks']['completeness'] = False
            results['passed'] = False
            return results
        results['checks']['completeness'] = True
        
        required_cols = ['alert_id', 'score', 'model_version', 'latency_ms']
        if not all(col in scores_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in scores_df.columns]
            logger.error(f"Schema check failed: missing columns {missing}")
            results['checks']['schema'] = False
            results['passed'] = False
            return results
        results['checks']['schema'] = True
        
        if not scores_df['score'].between(0, 1).all():
            invalid = scores_df[~scores_df['score'].between(0, 1)]
            logger.error(f"Score range check failed: {len(invalid)} scores out of [0, 1] range")
            results['checks']['score_range'] = False
            results['passed'] = False
            return results
        results['checks']['score_range'] = True
        
        avg_latency = scores_df['latency_ms'].mean()
        if avg_latency > self.max_latency_ms:
            logger.error(f"Latency check failed: {avg_latency:.2f}ms > {self.max_latency_ms}ms")
            results['checks']['latency'] = False
            results['passed'] = False
            return results
        results['checks']['latency'] = True
        
        alert_ids_match = set(scores_df['alert_id']) == set(alerts_df['alert_id'])
        if not alert_ids_match:
            logger.error("Alert ID mismatch between scores and input")
            results['checks']['alert_ids_match'] = False
            results['passed'] = False
            return results
        results['checks']['alert_ids_match'] = True
        
        results['score'] = 0.2
        logger.info(f"Integrity validation passed: score={results['score']:.3f}")
        
        return results
    
    def validate_determinism(self, scores_df_1: pd.DataFrame, scores_df_2: pd.DataFrame) -> Dict:
        results = {
            'passed': True,
            'differences': []
        }
        
        if len(scores_df_1) != len(scores_df_2):
            results['passed'] = False
            results['differences'].append('different_lengths')
            return results
        
        merged = scores_df_1.merge(
            scores_df_2,
            on='alert_id',
            suffixes=('_1', '_2'),
            how='outer'
        )
        
        score_diff = (merged['score_1'] - merged['score_2']).abs()
        non_deterministic = merged[score_diff > 1e-6]
        
        if len(non_deterministic) > 0:
            results['passed'] = False
            results['differences'].append(f'{len(non_deterministic)}_non_deterministic_scores')
            logger.error(f"Determinism check failed: {len(non_deterministic)} alerts with different scores")
        
        return results