from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list


class ScoresRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "alert_scores.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "alert_scores"
    
    def insert_scores(self, scores_df: pd.DataFrame, processing_date: str, network: str):
        scores_df = scores_df.copy()
        scores_df['processing_date'] = processing_date
        scores_df['network'] = network
        
        required_columns = ['processing_date', 'network', 'alert_id', 'score', 'model_version', 'latency_ms']
        if not all(col in scores_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in scores_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        if 'explain_json' not in scores_df.columns:
            scores_df['explain_json'] = ''
        
        self.client.insert_df(
            self.table_name(),
            scores_df[required_columns + ['explain_json']]
        )
        
        logger.info(f"Inserted {len(scores_df)} scores for {processing_date}/{network}")
    
    def get_scores(self, processing_date: str, network: str) -> List[AlertScore]:
        query = f'''
            SELECT processing_date, network, alert_id, score, model_version, latency_ms, explain_json
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY score DESC
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        return rows_to_pydantic_list(
            AlertScore,
            result.result_rows,
            result.column_names
        )
    
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
    
    def get_available_dates(self, network: str) -> List[str]:
        query = f'''
            SELECT DISTINCT processing_date
            FROM {self.table_name()}
            WHERE network = %(network)s
            ORDER BY processing_date DESC
        '''
        
        result = self.client.query(query, {'network': network})
        
        return [str(row[0]) for row in result.result_rows]
    
    def delete_old_scores(self, retention_days: int, network: str):
        query = f'''
            ALTER TABLE {self.table_name()}
            DELETE WHERE network = %(network)s 
            AND processing_date < today() - INTERVAL %(days)s DAY
        '''
        
        self.client.command(query, {'network': network, 'days': retention_days})
        
        logger.info(f"Deleted scores older than {retention_days} days for {network}")