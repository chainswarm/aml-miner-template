from typing import List, Optional
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list


class FeaturesRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "raw_features.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "raw_features"
    
    def insert_features(self, features_df: pd.DataFrame, processing_date: str, network: str):
        features_df = features_df.copy()
        features_df['processing_date'] = processing_date
        features_df['network'] = network
        
        required_columns = ['processing_date', 'network', 'address', 'feature_name', 'feature_value']
        if not all(col in features_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in features_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        if 'feature_metadata' not in features_df.columns:
            features_df['feature_metadata'] = ''
        
        self.client.insert_df(
            self.table_name(),
            features_df[required_columns + ['feature_metadata']]
        )
        
        logger.info(f"Inserted {len(features_df)} features for {processing_date}/{network}")
    
    def get_features(self, processing_date: str, network: str) -> List[Feature]:
        query = f'''
            SELECT processing_date, network, address, feature_name, feature_value, feature_metadata
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY address, feature_name
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        
        return rows_to_pydantic_list(
            Feature,
            result.result_rows,
            result.column_names
        )
    
    def get_features_for_alert(self, alert_id: str, processing_date: str, network: str) -> List[Feature]:
        query = f'''
            SELECT f.processing_date, f.network, f.address, f.feature_name, f.feature_value, f.feature_metadata
            FROM {self.table_name()} f
            INNER JOIN raw_alerts a ON f.address = a.address 
                AND f.processing_date = a.processing_date 
                AND f.network = a.network
            WHERE a.alert_id = %(alert_id)s 
                AND f.processing_date = %(date)s 
                AND f.network = %(network)s
            ORDER BY f.feature_name
        '''
        
        result = self.client.query(query, {
            'alert_id': alert_id,
            'date': processing_date,
            'network': network
        })
        
        return rows_to_pydantic_list(
            Feature,
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