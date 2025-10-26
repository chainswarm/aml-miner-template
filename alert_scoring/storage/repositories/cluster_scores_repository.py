from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list


class ClusterScoreData(BaseModel):
    processing_date: str
    network: str
    cluster_id: str
    score: float
    model_version: str


class ClusterScoresRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "cluster_scores.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "cluster_scores"
    
    def insert_scores(self, scores_df: pd.DataFrame, processing_date: str, network: str):
        scores_df = scores_df.copy()
        scores_df['processing_date'] = processing_date
        scores_df['network'] = network
        
        required_columns = ['processing_date', 'network', 'cluster_id', 'score', 'model_version']
        if not all(col in scores_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in scores_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        self.client.insert_df(
            self.table_name(),
            scores_df[required_columns]
        )
        
        logger.info(f"Inserted {len(scores_df)} cluster scores for {processing_date}/{network}")
    
    def get_scores(self, processing_date: str, network: str) -> List[ClusterScoreData]:
        query = f'''
            SELECT processing_date, network, cluster_id, score, model_version
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY cluster_id
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        return rows_to_pydantic_list(
            ClusterScoreData,
            result.result_rows,
            result.column_names
        )
    
    def get_score_by_id(self, cluster_id: str, processing_date: str, network: str) -> Optional[ClusterScoreData]:
        query = f'''
            SELECT processing_date, network, cluster_id, score, model_version
            FROM {self.table_name()}
            WHERE cluster_id = %(cluster_id)s 
                AND processing_date = %(date)s 
                AND network = %(network)s
            LIMIT 1
        '''
        
        result = self.client.query(query, {
            'cluster_id': cluster_id,
            'date': processing_date,
            'network': network
        })
        
        scores = rows_to_pydantic_list(
            ClusterScoreData,
            result.result_rows,
            result.column_names
        )
        
        return scores[0] if scores else None
    
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