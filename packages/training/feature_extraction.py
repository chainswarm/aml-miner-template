from typing import Dict
import pandas as pd
from clickhouse_connect.driver import Client
from loguru import logger


class FeatureExtractor:
    
    def __init__(self, client: Client):
        self.client = client
    
    def extract_training_data(
        self,
        start_date: str,
        end_date: str,
        window_days: int = 7
    ) -> Dict[str, pd.DataFrame]:
        
        logger.info(
            "Extracting training data",
            extra={
                "start_date": start_date,
                "end_date": end_date,
                "window_days": window_days
            }
        )
        
        data = {
            'alerts': self._extract_alerts(start_date, end_date, window_days),
            'features': self._extract_features(start_date, end_date, window_days),
            'clusters': self._extract_clusters(start_date, end_date, window_days),
            'money_flows': self._extract_money_flows(start_date, end_date, window_days),
            'address_labels': self._extract_address_labels(start_date, end_date, window_days)
        }
        
        self._validate_extracted_data(data)
        
        logger.success(
            "Data extraction completed",
            extra={
                "alerts": len(data['alerts']),
                "features": len(data['features']),
                "clusters": len(data['clusters']),
                "money_flows": len(data['money_flows']),
                "address_labels": len(data['address_labels'])
            }
        )
        
        return data
    
    def _extract_alerts(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        
        query = f"""
            SELECT
                alert_id,
                address,
                processing_date,
                window_days,
                typology_type,
                pattern_id,
                pattern_type,
                severity,
                suspected_address_type,
                suspected_address_subtype,
                alert_confidence_score,
                description,
                volume_usd,
                evidence_json,
                risk_indicators
            FROM raw_alerts
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, alert_id
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            raise ValueError(
                f"No alerts found for processing_date {start_date} to {end_date} "
                f"with window_days={window_days}"
            )
        
        df = pd.DataFrame(
            result.result_rows,
            columns=result.column_names
        )
        
        num_snapshots = df['processing_date'].nunique()
        logger.info(
            f"Extracted {len(df):,} alerts from {num_snapshots} snapshots"
        )
        
        return df
    
    def _extract_features(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        
        query = f"""
            SELECT
                address,
                processing_date,
                window_days,
                total_in_usd,
                total_out_usd,
                tx_total_count,
                unique_counterparties,
                avg_tx_in_usd,
                avg_tx_out_usd,
                max_tx_usd,
                is_exchange_like,
                is_mixer_like,
                behavioral_anomaly_score,
                graph_anomaly_score,
                global_anomaly_score
            FROM raw_features
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, address
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            raise ValueError(
                f"No features found for processing_date {start_date} to {end_date} "
                f"with window_days={window_days}"
            )
        
        df = pd.DataFrame(
            result.result_rows,
            columns=result.column_names
        )
        
        logger.info(f"Extracted {len(df):,} feature records")
        
        return df
    
    def _extract_clusters(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        
        query = f"""
            SELECT
                cluster_id,
                processing_date,
                window_days,
                cluster_type,
                total_alerts,
                total_volume_usd,
                addresses_involved,
                severity_max,
                confidence_avg
            FROM raw_clusters
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, cluster_id
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            logger.warning("No clusters found - this is optional")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            result.result_rows,
            columns=result.column_names
        )
        
        logger.info(f"Extracted {len(df):,} clusters")
        
        return df
    
    def _extract_money_flows(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        
        query = f"""
            SELECT
                from_address,
                to_address,
                processing_date,
                window_days,
                tx_count,
                amount_usd_sum,
                first_seen_timestamp,
                last_seen_timestamp,
                avg_tx_size_usd,
                is_bidirectional
            FROM raw_money_flows
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, from_address, to_address
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            logger.warning("No money flows found - this is optional")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            result.result_rows,
            columns=result.column_names
        )
        
        logger.info(f"Extracted {len(df):,} money flows")
        
        return df
    
    def _extract_address_labels(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        
        query = f"""
            SELECT
                processing_date,
                window_days,
                network,
                address,
                label,
                address_type,
                address_subtype,
                risk_level,
                confidence_score,
                source
            FROM raw_address_labels
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, address
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            logger.warning("No address labels found - this is optional")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            result.result_rows,
            columns=result.column_names
        )
        
        logger.info(f"Extracted {len(df):,} address labels")
        
        return df
    
    def _validate_extracted_data(self, data: Dict[str, pd.DataFrame]):
        
        if data['alerts'].empty:
            raise ValueError("Alerts dataframe is empty")
        
        if data['features'].empty:
            raise ValueError("Features dataframe is empty")
        
        required_alert_cols = ['alert_id', 'address', 'processing_date']
        missing_cols = set(required_alert_cols) - set(data['alerts'].columns)
        if missing_cols:
            raise ValueError(f"Missing required alert columns: {missing_cols}")
        
        required_feature_cols = ['address', 'processing_date']
        missing_cols = set(required_feature_cols) - set(data['features'].columns)
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        logger.info("Data validation passed")