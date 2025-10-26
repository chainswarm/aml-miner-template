from fastapi import APIRouter, HTTPException
from typing import List
import pandas as pd
from loguru import logger
from datetime import datetime
import time
import json

from aml_miner.api.schemas import (
    BatchData,
    ScoreResponse,
    RankResponse,
    ClusterScoreResponse,
    HealthResponse,
    VersionResponse
)
from aml_miner.version import __version__

router = APIRouter()

alert_scorer = None
alert_ranker = None
cluster_scorer = None
settings = None


def set_models(scorer, ranker, cluster, cfg):
    global alert_scorer, alert_ranker, cluster_scorer, settings
    alert_scorer = scorer
    alert_ranker = ranker
    cluster_scorer = cluster
    settings = cfg


def convert_batch_to_dataframes(batch: BatchData):
    alerts_df = pd.DataFrame([alert.model_dump() for alert in batch.alerts])
    
    features_df = pd.DataFrame(batch.features)
    
    clusters_df = pd.DataFrame([cluster.model_dump() for cluster in batch.clusters])
    
    money_flows_df = None
    if batch.money_flows:
        money_flows_df = pd.DataFrame([flow.model_dump() for flow in batch.money_flows])
    
    return alerts_df, features_df, clusters_df, money_flows_df


@router.post("/score/alerts", response_model=List[ScoreResponse])
async def score_alerts(batch: BatchData):
    if alert_scorer is None or alert_scorer.model is None:
        raise HTTPException(status_code=503, detail="Alert scorer model not loaded")
    
    try:
        start_time = time.time()
        
        logger.info(f"Scoring {len(batch.alerts)} alerts")
        
        alerts_df, features_df, clusters_df, _ = convert_batch_to_dataframes(batch)
        
        X = alert_scorer.prepare_features(alerts_df, features_df, clusters_df)
        
        scores = alert_scorer.predict(X)
        
        latency_ms = (time.time() - start_time) * 1000
        avg_latency = latency_ms / len(batch.alerts) if batch.alerts else 0
        
        model_version = getattr(alert_scorer, 'version', '1.0.0')
        
        responses = []
        for i, alert in enumerate(batch.alerts):
            explain_data = {
                "alert_id": alert.alert_id,
                "top_features": ["volume_usd", "alert_confidence_score"],
                "feature_contributions": {}
            }
            
            response = ScoreResponse(
                alert_id=alert.alert_id,
                score=float(scores[i]),
                model_version=model_version,
                latency_ms=round(avg_latency, 2),
                explain_json=json.dumps(explain_data)
            )
            responses.append(response)
        
        logger.info(f"Scored {len(responses)} alerts in {latency_ms:.2f}ms")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error scoring alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scoring alerts: {str(e)}")


@router.post("/rank/alerts", response_model=List[RankResponse])
async def rank_alerts(batch: BatchData):
    if alert_ranker is None or alert_ranker.model is None:
        raise HTTPException(status_code=503, detail="Alert ranker model not loaded")
    
    try:
        start_time = time.time()
        
        logger.info(f"Ranking {len(batch.alerts)} alerts")
        
        alerts_df, features_df, clusters_df, _ = convert_batch_to_dataframes(batch)
        
        X = alert_ranker.prepare_features(alerts_df, features_df, clusters_df)
        
        alert_ids = [alert.alert_id for alert in batch.alerts]
        
        ranking_df = alert_ranker.rank_alerts(X, alert_ids)
        
        latency_ms = (time.time() - start_time) * 1000
        
        responses = []
        for _, row in ranking_df.iterrows():
            response = RankResponse(
                alert_id=row['alert_id'],
                rank=int(row['rank']),
                score=float(row['score'])
            )
            responses.append(response)
        
        logger.info(f"Ranked {len(responses)} alerts in {latency_ms:.2f}ms")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error ranking alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error ranking alerts: {str(e)}")


@router.post("/score/clusters", response_model=List[ClusterScoreResponse])
async def score_clusters(batch: BatchData):
    if cluster_scorer is None or cluster_scorer.model is None:
        raise HTTPException(status_code=503, detail="Cluster scorer model not loaded")
    
    try:
        start_time = time.time()
        
        logger.info(f"Scoring {len(batch.clusters)} clusters")
        
        alerts_df, features_df, clusters_df, _ = convert_batch_to_dataframes(batch)
        
        X = cluster_scorer.prepare_features(alerts_df, features_df, clusters_df)
        
        scores = cluster_scorer.predict(X)
        
        latency_ms = (time.time() - start_time) * 1000
        avg_latency = latency_ms / len(batch.clusters) if batch.clusters else 0
        
        model_version = getattr(cluster_scorer, 'version', '1.0.0')
        
        responses = []
        for i, cluster in enumerate(batch.clusters):
            explain_data = {
                "cluster_id": cluster.cluster_id,
                "top_features": ["total_alerts", "severity_max"],
                "feature_contributions": {}
            }
            
            response = ClusterScoreResponse(
                cluster_id=cluster.cluster_id,
                score=float(scores[i]),
                model_version=model_version,
                latency_ms=round(avg_latency, 2),
                explain_json=json.dumps(explain_data)
            )
            responses.append(response)
        
        logger.info(f"Scored {len(responses)} clusters in {latency_ms:.2f}ms")
        
        return responses
        
    except Exception as e:
        logger.error(f"Error scoring clusters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error scoring clusters: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    try:
        models_loaded = {
            "alert_scorer": alert_scorer is not None and alert_scorer.model is not None,
            "alert_ranker": alert_ranker is not None and alert_ranker.model is not None,
            "cluster_scorer": cluster_scorer is not None and cluster_scorer.model is not None
        }
        
        all_loaded = all(models_loaded.values())
        status = "healthy" if all_loaded else "degraded"
        
        return HealthResponse(
            status=status,
            models_loaded=models_loaded,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
        
    except Exception as e:
        logger.error(f"Error in health check: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in health check: {str(e)}")


@router.get("/version", response_model=VersionResponse)
async def get_version():
    try:
        model_versions = {
            "alert_scorer": getattr(alert_scorer, 'version', 'unknown') if alert_scorer else 'not_loaded',
            "alert_ranker": getattr(alert_ranker, 'version', 'unknown') if alert_ranker else 'not_loaded',
            "cluster_scorer": getattr(cluster_scorer, 'version', 'unknown') if cluster_scorer else 'not_loaded'
        }
        
        return VersionResponse(
            api_version=__version__,
            model_versions=model_versions
        )
        
    except Exception as e:
        logger.error(f"Error getting version: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting version: {str(e)}")