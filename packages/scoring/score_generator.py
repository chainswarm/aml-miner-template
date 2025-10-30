import time
import pandas as pd
import numpy as np
from typing import Union, Tuple
import xgboost as xgb
import lightgbm as lgb
from loguru import logger


class ScoreGenerator:
    
    def score_alerts(
        self,
        model: Union[xgb.XGBClassifier, lgb.Booster],
        features: pd.DataFrame,
        alert_ids: pd.Series
    ) -> Tuple[pd.DataFrame, float]:
        
        start_time = time.time()
        
        if isinstance(model, xgb.XGBClassifier):
            scores = model.predict_proba(features)[:, 1]
        else:
            scores = model.predict(features)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = pd.DataFrame({
            'alert_id': alert_ids.values,
            'score': scores
        })
        
        logger.info(
            f"Scored {len(result)} alerts in {latency_ms:.2f}ms "
            f"({latency_ms/len(result):.2f}ms per alert)"
        )
        
        return result, latency_ms
    
    def rank_alerts(
        self,
        model: Union[xgb.XGBClassifier, lgb.Booster],
        features: pd.DataFrame,
        alert_ids: pd.Series
    ) -> Tuple[pd.DataFrame, float]:
        
        start_time = time.time()
        
        if isinstance(model, xgb.XGBClassifier):
            scores = model.predict_proba(features)[:, 1]
        else:
            scores = model.predict(features)
        
        rankings = pd.DataFrame({
            'alert_id': alert_ids.values,
            'score': scores
        })
        
        rankings = rankings.sort_values('score', ascending=False).reset_index(drop=True)
        rankings['rank'] = range(1, len(rankings) + 1)
        rankings = rankings[['alert_id', 'rank']]
        
        latency_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Ranked {len(rankings)} alerts in {latency_ms:.2f}ms"
        )
        
        return rankings, latency_ms
    
    def score_clusters(
        self,
        model: Union[xgb.XGBClassifier, lgb.Booster],
        features: pd.DataFrame,
        cluster_ids: pd.Series
    ) -> Tuple[pd.DataFrame, float]:
        
        start_time = time.time()
        
        if isinstance(model, xgb.XGBClassifier):
            scores = model.predict_proba(features)[:, 1]
        else:
            scores = model.predict(features)
        
        latency_ms = (time.time() - start_time) * 1000
        
        result = pd.DataFrame({
            'cluster_id': cluster_ids.values,
            'score': scores
        })
        
        logger.info(
            f"Scored {len(result)} clusters in {latency_ms:.2f}ms"
        )
        
        return result, latency_ms