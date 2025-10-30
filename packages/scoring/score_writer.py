from datetime import datetime
from typing import Dict, Any
import pandas as pd
from clickhouse_connect.driver import Client
from loguru import logger


class ScoreWriter:
    
    def __init__(self, client: Client):
        self.client = client
    
    def write_alert_scores(
        self,
        processing_date: str,
        scores: pd.DataFrame,
        model_version: str,
        latency_ms: float
    ):
        
        if scores.empty:
            logger.warning("No alert scores to write")
            return
        
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        rows = []
        for _, row in scores.iterrows():
            rows.append([
                date_obj,
                row['alert_id'],
                float(row['score']),
                model_version,
                float(latency_ms),
                '',
                datetime.utcnow()
            ])
        
        self.client.insert(
            'alert_scores',
            rows,
            column_names=[
                'processing_date', 'alert_id', 'score', 'model_version',
                'latency_ms', 'explain_json', 'created_at'
            ]
        )
        
        logger.info(f"Wrote {len(rows)} alert scores to ClickHouse")
    
    def write_alert_rankings(
        self,
        processing_date: str,
        rankings: pd.DataFrame,
        model_version: str
    ):
        
        if rankings.empty:
            logger.warning("No alert rankings to write")
            return
        
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        rows = []
        for _, row in rankings.iterrows():
            rows.append([
                date_obj,
                row['alert_id'],
                int(row['rank']),
                model_version,
                datetime.utcnow()
            ])
        
        self.client.insert(
            'alert_rankings',
            rows,
            column_names=[
                'processing_date', 'alert_id', 'rank', 
                'model_version', 'created_at'
            ]
        )
        
        logger.info(f"Wrote {len(rows)} alert rankings to ClickHouse")
    
    def write_cluster_scores(
        self,
        processing_date: str,
        scores: pd.DataFrame,
        model_version: str
    ):
        
        if scores.empty:
            logger.warning("No cluster scores to write")
            return
        
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        rows = []
        for _, row in scores.iterrows():
            rows.append([
                date_obj,
                row['cluster_id'],
                float(row['score']),
                model_version,
                datetime.utcnow()
            ])
        
        self.client.insert(
            'cluster_scores',
            rows,
            column_names=[
                'processing_date', 'cluster_id', 'score',
                'model_version', 'created_at'
            ]
        )
        
        logger.info(f"Wrote {len(rows)} cluster scores to ClickHouse")
    
    def update_batch_metadata(
        self,
        processing_date: str,
        metadata: Dict[str, Any]
    ):
        
        date_obj = datetime.strptime(processing_date, '%Y-%m-%d').date()
        
        row = [[
            date_obj,
            datetime.utcnow(),
            metadata.get('input_counts_alerts', 0),
            metadata.get('input_counts_features', 0),
            metadata.get('input_counts_clusters', 0),
            metadata.get('output_counts_alert_scores', 0),
            metadata.get('output_counts_alert_rankings', 0),
            metadata.get('output_counts_cluster_scores', 0),
            int(metadata.get('latencies_ms_alert_scoring', 0)),
            int(metadata.get('latencies_ms_alert_ranking', 0)),
            int(metadata.get('latencies_ms_cluster_scoring', 0)),
            int(metadata.get('latencies_ms_total', 0)),
            metadata.get('model_versions_alert_scorer', ''),
            metadata.get('model_versions_alert_ranker', ''),
            metadata.get('model_versions_cluster_scorer', ''),
            metadata.get('status', 'COMPLETED'),
            metadata.get('error_message', ''),
            datetime.utcnow()
        ]]
        
        self.client.insert(
            'batch_metadata',
            row,
            column_names=[
                'processing_date', 'processed_at',
                'input_counts_alerts', 'input_counts_features', 'input_counts_clusters',
                'output_counts_alert_scores', 'output_counts_alert_rankings', 
                'output_counts_cluster_scores',
                'latencies_ms_alert_scoring', 'latencies_ms_alert_ranking',
                'latencies_ms_cluster_scoring', 'latencies_ms_total',
                'model_versions_alert_scorer', 'model_versions_alert_ranker',
                'model_versions_cluster_scorer',
                'status', 'error_message', 'created_at'
            ]
        )
        
        logger.success("Batch metadata updated in ClickHouse")