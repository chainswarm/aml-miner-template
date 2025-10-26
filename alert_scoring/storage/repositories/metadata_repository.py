from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list


class BatchMetadata(BaseModel):
    processing_date: str
    network: str
    processed_at: datetime
    input_counts_alerts: int
    input_counts_features: int
    input_counts_clusters: int
    output_counts_alert_scores: int
    output_counts_alert_rankings: int
    output_counts_cluster_scores: int
    latencies_ms_alert_scoring: int
    latencies_ms_alert_ranking: int
    latencies_ms_cluster_scoring: int
    latencies_ms_total: int
    model_versions_alert_scorer: str
    model_versions_alert_ranker: str
    model_versions_cluster_scorer: str
    status: str = "PROCESSING"
    error_message: str = ""


class MetadataRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "batch_metadata.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "batch_metadata"
    
    def insert_metadata(self, processing_date: str, network: str, batch_info: Dict[str, Any]):
        metadata = {
            'processing_date': processing_date,
            'network': network,
            'processed_at': batch_info.get('processed_at', datetime.utcnow()),
            'input_counts_alerts': batch_info.get('input_counts', {}).get('alerts', 0),
            'input_counts_features': batch_info.get('input_counts', {}).get('features', 0),
            'input_counts_clusters': batch_info.get('input_counts', {}).get('clusters', 0),
            'output_counts_alert_scores': batch_info.get('output_counts', {}).get('alert_scores', 0),
            'output_counts_alert_rankings': batch_info.get('output_counts', {}).get('alert_rankings', 0),
            'output_counts_cluster_scores': batch_info.get('output_counts', {}).get('cluster_scores', 0),
            'latencies_ms_alert_scoring': batch_info.get('latencies_ms', {}).get('alert_scoring', 0),
            'latencies_ms_alert_ranking': batch_info.get('latencies_ms', {}).get('alert_ranking', 0),
            'latencies_ms_cluster_scoring': batch_info.get('latencies_ms', {}).get('cluster_scoring', 0),
            'latencies_ms_total': batch_info.get('latencies_ms', {}).get('total', 0),
            'model_versions_alert_scorer': batch_info.get('model_versions', {}).get('alert_scorer', ''),
            'model_versions_alert_ranker': batch_info.get('model_versions', {}).get('alert_ranker', ''),
            'model_versions_cluster_scorer': batch_info.get('model_versions', {}).get('cluster_scorer', ''),
            'status': batch_info.get('status', 'PROCESSING'),
            'error_message': batch_info.get('error_message', '')
        }
        
        query = f'''
            INSERT INTO {self.table_name()} (
                processing_date, network, processed_at,
                input_counts_alerts, input_counts_features, input_counts_clusters,
                output_counts_alert_scores, output_counts_alert_rankings, output_counts_cluster_scores,
                latencies_ms_alert_scoring, latencies_ms_alert_ranking, latencies_ms_cluster_scoring, latencies_ms_total,
                model_versions_alert_scorer, model_versions_alert_ranker, model_versions_cluster_scorer,
                status, error_message
            ) VALUES (
                %(processing_date)s, %(network)s, %(processed_at)s,
                %(input_counts_alerts)s, %(input_counts_features)s, %(input_counts_clusters)s,
                %(output_counts_alert_scores)s, %(output_counts_alert_rankings)s, %(output_counts_cluster_scores)s,
                %(latencies_ms_alert_scoring)s, %(latencies_ms_alert_ranking)s, %(latencies_ms_cluster_scoring)s, %(latencies_ms_total)s,
                %(model_versions_alert_scorer)s, %(model_versions_alert_ranker)s, %(model_versions_cluster_scorer)s,
                %(status)s, %(error_message)s
            )
        '''
        
        self.client.command(query, metadata)
        
        logger.info(f"Inserted metadata for {processing_date}/{network}")
    
    def get_metadata(self, processing_date: str, network: str) -> Optional[BatchMetadata]:
        query = f'''
            SELECT processing_date, network, processed_at,
                   input_counts_alerts, input_counts_features, input_counts_clusters,
                   output_counts_alert_scores, output_counts_alert_rankings, output_counts_cluster_scores,
                   latencies_ms_alert_scoring, latencies_ms_alert_ranking, latencies_ms_cluster_scoring, latencies_ms_total,
                   model_versions_alert_scorer, model_versions_alert_ranker, model_versions_cluster_scorer,
                   status, error_message
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY processed_at DESC
            LIMIT 1
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        metadata_list = rows_to_pydantic_list(
            BatchMetadata,
            result.result_rows,
            result.column_names
        )
        
        return metadata_list[0] if metadata_list else None
    
    def get_latest_metadata(self, network: str) -> Optional[BatchMetadata]:
        query = f'''
            SELECT processing_date, network, processed_at,
                   input_counts_alerts, input_counts_features, input_counts_clusters,
                   output_counts_alert_scores, output_counts_alert_rankings, output_counts_cluster_scores,
                   latencies_ms_alert_scoring, latencies_ms_alert_ranking, latencies_ms_cluster_scoring, latencies_ms_total,
                   model_versions_alert_scorer, model_versions_alert_ranker, model_versions_cluster_scorer,
                   status, error_message
            FROM {self.table_name()}
            WHERE network = %(network)s
            ORDER BY processing_date DESC, processed_at DESC
            LIMIT 1
        '''
        
        result = self.client.query(query, {'network': network})
        
        metadata_list = rows_to_pydantic_list(
            BatchMetadata,
            result.result_rows,
            result.column_names
        )
        
        return metadata_list[0] if metadata_list else None
    
    def update_status(self, processing_date: str, network: str, status: str, error_message: str = ""):
        query = f'''
            ALTER TABLE {self.table_name()}
            UPDATE status = %(status)s, error_message = %(error_message)s
            WHERE processing_date = %(date)s AND network = %(network)s
        '''
        
        self.client.command(query, {
            'date': processing_date,
            'network': network,
            'status': status,
            'error_message': error_message
        })
        
        logger.info(f"Updated status to {status} for {processing_date}/{network}")