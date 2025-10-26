from typing import Dict, List
import pandas as pd
import numpy as np
from loguru import logger


class BehaviorValidator:
    def __init__(self, min_variance: float = 0.001):
        self.min_variance = min_variance
    
    def validate(self, scores_df: pd.DataFrame, pattern_traps: List[Dict]) -> Dict:
        results = {
            'passed': True,
            'score': 0.0,
            'traps_detected': [],
            'checks': {}
        }
        
        score_variance = scores_df['score'].var()
        if score_variance < self.min_variance:
            results['traps_detected'].append('constant_scores')
            results['checks']['variance'] = False
            results['score'] -= 0.1
            logger.warning(f"Low variance detected: {score_variance:.6f}")
        else:
            results['checks']['variance'] = True
        
        for trap in pattern_traps:
            trap_alert_id = trap['alert_id']
            expected_score = trap['expected_score']
            
            if trap_alert_id not in scores_df['alert_id'].values:
                continue
            
            actual_score = scores_df[scores_df['alert_id'] == trap_alert_id]['score'].values[0]
            
            if abs(actual_score - expected_score) < 0.01:
                results['traps_detected'].append(f"trap_{trap_alert_id}")
                results['score'] -= 0.05
                logger.warning(f"Pattern trap detected: {trap_alert_id}")
        
        results['checks']['pattern_traps'] = len(results['traps_detected']) == 0
        
        median_score = scores_df['score'].median()
        if median_score < 0.1 or median_score > 0.9:
            results['traps_detected'].append('extreme_median')
            results['checks']['median_reasonable'] = False
            results['score'] -= 0.05
            logger.warning(f"Extreme median score: {median_score:.3f}")
        else:
            results['checks']['median_reasonable'] = True
        
        results['score'] = max(0.0, min(0.3, 0.3 + results['score']))
        
        if len(results['traps_detected']) > 0:
            logger.warning(f"Behavior validation issues: {results['traps_detected']}")
        else:
            logger.info(f"Behavior validation passed: score={results['score']:.3f}")
        
        return results
    
    def detect_plagiarism(self, scores_df: pd.DataFrame, other_scores: List[pd.DataFrame]) -> Dict:
        results = {
            'plagiarism_detected': False,
            'similar_to': [],
            'similarity_scores': []
        }
        
        for i, other_df in enumerate(other_scores):
            merged = scores_df.merge(
                other_df,
                on='alert_id',
                suffixes=('_self', '_other'),
                how='inner'
            )
            
            if len(merged) < 10:
                continue
            
            correlation = merged['score_self'].corr(merged['score_other'])
            
            if correlation > 0.95:
                results['plagiarism_detected'] = True
                results['similar_to'].append(i)
                results['similarity_scores'].append(correlation)
                logger.error(f"High similarity ({correlation:.3f}) with submission {i}")
        
        return results