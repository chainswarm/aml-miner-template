from typing import List, Optional, Dict
import pandas as pd
from pydantic import BaseModel
from loguru import logger

from alert_scoring.storage.repositories.base_repository import BaseRepository
from alert_scoring.storage.utils import rows_to_pydantic_list, row_to_dict


class AlertsRepository(BaseRepository):
    @classmethod
    def schema(cls) -> str:
        return "raw_alerts.sql"
    
    @classmethod
    def table_name(cls) -> str:
        return "raw_alerts"
    
    def insert_alerts(self, alerts_df: pd.DataFrame, processing_date: str, network: str):
        alerts_df = alerts_df.copy()
        alerts_df['processing_date'] = processing_date
        alerts_df['network'] = network
        
        required_columns = [
            'window_days', 'processing_date', 'network', 'alert_id', 'address',
            'typology_type', 'alert_confidence_score', 'description', 'evidence_json',
            'risk_indicators'
        ]
        if not all(col in alerts_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in alerts_df.columns]
            raise ValueError(f"Missing required columns: {missing}")
        
        optional_columns = ['pattern_id', 'pattern_type', 'severity', 'suspected_address_type', 
                          'suspected_address_subtype', 'volume_usd']
        for col in optional_columns:
            if col not in alerts_df.columns:
                if col == 'pattern_id' or col == 'pattern_type' or col == 'suspected_address_subtype':
                    alerts_df[col] = ''
                elif col == 'severity':
                    alerts_df[col] = 'MEDIUM'
                elif col == 'suspected_address_type':
                    alerts_df[col] = 'unknown'
                elif col == 'volume_usd':
                    alerts_df[col] = 0
        
        columns_to_insert = required_columns + optional_columns
        self.client.insert_df(
            self.table_name(),
            alerts_df[columns_to_insert]
        )
        
        logger.info(f"Inserted {len(alerts_df)} alerts for {processing_date}/{network}")
    
    def get_alerts(self, processing_date: str, network: str) -> List[Dict]:
        query = f'''
            SELECT window_days, processing_date, network, alert_id, address, typology_type,
                   pattern_id, pattern_type, severity, suspected_address_type, 
                   suspected_address_subtype, alert_confidence_score, description, 
                   volume_usd, evidence_json, risk_indicators
            FROM {self.table_name()}
            WHERE processing_date = %(date)s AND network = %(network)s
            ORDER BY alert_id
        '''
        
        result = self.client.query(query, {'date': processing_date, 'network': network})
        return [row in row_to_dict(row, result.column_names) for row in result.result_rows]
    
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