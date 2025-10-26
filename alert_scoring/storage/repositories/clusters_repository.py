from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list


class Cluster(BaseModel):
    window_days: int
    processing_date: str
    network: str
    cluster_id: str
    cluster_type: str
    primary_address: str = ""
    pattern_id: str = ""
    primary_alert_id: str
    related_alert_ids: List[str]
    addresses_involved: List[str]
    total_alerts: int
    total_volume_usd: float
    severity_max: str = "MEDIUM"
    confidence_avg: float
    earliest_alert_timestamp: int
    latest_alert_timestamp: int


class ClustersRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "raw_clusters.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "raw_clusters"
    
    def insert_clusters(self, clusters_df: pd.DataFrame, processing_date: str, network: str):
        clusters_df = clusters_df.copy()
        clusters_df['processing_date'] = processing_date
        clusters_df['network'] = network
        
        required_columns = [
            'window_days', 'processing_date', 'network', 'cluster_id', 'cluster_type',
            'primary_alert_id', 'related_alert_ids', 'addresses_involved', 'total_alerts',
            'total_volume_usd', 'confidence_avg', 'earliest_alert_timestamp', 
            'latest_alert_timestamp'
        ]
        if not all(col in clusters_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in clusters_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        optional_columns = ['primary_address', 'pattern_id', 'severity_max']
        for col in optional_columns:
            if col not in clusters_df.columns:
                if col == 'primary_address' or col == 'pattern_id':
                    clusters_df[col] = ''
                elif col == 'severity_max':
                    clusters_df[col] = 'MEDIUM'
        
        columns_to_insert = required_columns + optional_columns
        self.client.insert_df(
            self.table_name(),
            clusters_df[columns_to_insert]
        )
        
        logger.info(f"Inserted {len(clusters_df)} clusters for {processing_date}/{network}")
    
    def get_clusters(self, processing_date: str, network: str) -> List[Cluster]:
        query = f'''
            SELECT window_days, processing_date, network, cluster_id, cluster_type,
                   primary_address, pattern_id, primary_alert_id, related_alert_ids,
                   addresses_involved, total_alerts, total_volume_usd, severity_max,
                   confidence_avg, earliest_alert_timestamp, latest_alert_timestamp
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY cluster_id
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        return rows_to_pydantic_list(
            Cluster,
            result.result_rows,
            result.column_names
        )
    
    def get_cluster_by_id(self, cluster_id: str, processing_date: str, network: str) -> Optional[Cluster]:
        query = f'''
            SELECT window_days, processing_date, network, cluster_id, cluster_type,
                   primary_address, pattern_id, primary_alert_id, related_alert_ids,
                   addresses_involved, total_alerts, total_volume_usd, severity_max,
                   confidence_avg, earliest_alert_timestamp, latest_alert_timestamp
            FROM {self.table_name()}
            WHERE cluster_id = %(cluster_id)s 
                AND processing_date = %(date)s 
                AND network = %(network)s
            ORDER BY cluster_id
            LIMIT 1
        '''
        
        result = self.client.query(query, {
            'cluster_id': cluster_id,
            'date': processing_date,
            'network': network
        })
        
        clusters = rows_to_pydantic_list(
            Cluster,
            result.result_rows,
            result.column_names
        )
        
        return clusters[0] if clusters else None
    
    def get_latest_date(self, network: str) -> Optional[str]:
        query = f'''
            SELECT max(processing_date) as latest
            FROM {self.table_name()}
            WHERE network = %(network)s
        '''
        
        result = self.client.query(query, {'network': network})
        
        if result.result_rows and result.result_rows[0][0]:
            return str(result.result_rows[0][0])
        
        return None