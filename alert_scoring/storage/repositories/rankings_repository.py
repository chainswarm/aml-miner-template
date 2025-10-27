from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list


class RankingsRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "alert_rankings.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "alert_rankings"
    
    def insert_rankings(self, rankings_df: pd.DataFrame, processing_date: str, network: str):
        rankings_df = rankings_df.copy()
        rankings_df['processing_date'] = processing_date
        rankings_df['network'] = network
        
        required_columns = ['processing_date', 'network', 'alert_id', 'rank', 'model_version']
        if not all(col in rankings_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in rankings_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        self.client.insert_df(
            self.table_name(),
            rankings_df[required_columns]
        )
        
        logger.info(f"Inserted {len(rankings_df)} rankings for {processing_date}/{network}")
    
    def get_rankings(self, processing_date: str, network: str) -> List[AlertRanking]:
        query = f'''
            SELECT processing_date, network, alert_id, rank, model_version
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY rank
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        return rows_to_pydantic_list(
            AlertRanking,
            result.result_rows,
            result.column_names
        )
    
    def get_top_n(self, n: int, processing_date: str, network: str) -> List[AlertRanking]:
        query = f'''
            SELECT processing_date, network, alert_id, rank, model_version
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY rank
            LIMIT %(limit)s
        '''
        
        result = self.client.query(query, {
            'date': processing_date,
            'network': network,
            'limit': n
        })
        
        return rows_to_pydantic_list(
            AlertRanking,
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