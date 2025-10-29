import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger


class FeatureBuilder:
    
    def build_training_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        
        logger.info("Building training features")
        
        X = data['alerts'].copy()
        
        if 'label' not in X.columns and 'ground_truth' not in X.columns:
            raise ValueError(
                "No label column found in alerts. "
                "Need 'label' or 'ground_truth' for supervised learning"
            )
        
        y = X.get('label', X.get('ground_truth'))
        
        X = self._add_alert_features(X)
        X = self._add_address_features(X, data['features'])
        X = self._add_temporal_features(X)
        X = self._add_statistical_features(X)
        
        if not data['clusters'].empty:
            X = self._add_cluster_features(X, data['clusters'])
        
        if not data['money_flows'].empty:
            X = self._add_network_features(X, data['money_flows'])
        
        if not data['address_labels'].empty:
            X = self._add_label_features(X, data['address_labels'])
        
        X = self._finalize_features(X)
        
        logger.success(
            "Feature building completed",
            extra={
                "num_samples": len(X),
                "num_features": len(X.columns),
                "positive_rate": float(y.mean())
            }
        )
        
        return X, y
    
    def _add_alert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Adding alert-level features")
        
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        df['severity_encoded'] = df['severity'].map(severity_map).fillna(2)
        
        df['volume_usd_log'] = np.log1p(df['volume_usd'].astype(float))
        
        df['confidence_score'] = df['alert_confidence_score'].fillna(0.5)
        
        address_type_map = {
            'exchange': 1, 'mixer': 2, 'defi': 3, 
            'contract': 4, 'eoa': 5, 'unknown': 0
        }
        df['address_type_encoded'] = df['suspected_address_type'].map(address_type_map).fillna(0)
        
        return df
    
    def _add_address_features(
        self,
        alerts_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Adding address-level features")
        
        merged = alerts_df.merge(
            features_df,
            on=['address', 'processing_date', 'window_days'],
            how='left',
            suffixes=('', '_feat')
        )
        
        merged['total_volume'] = (
            merged['total_received_usd'].fillna(0) +
            merged['total_sent_usd'].fillna(0)
        )
        
        merged['volume_ratio'] = (
            merged['total_sent_usd'] /
            (merged['total_received_usd'] + 1.0)
        )
        
        merged['is_exchange_flag'] = merged['is_exchange'].fillna(False).astype(int)
        merged['is_mixer_flag'] = merged['is_mixer'].fillna(False).astype(int)
        
        merged['address_risk_score'] = merged['risk_score'].fillna(0.0)
        
        merged['transaction_count_norm'] = merged['transaction_count'].fillna(0) / 100.0
        merged['unique_counterparties_norm'] = merged['unique_counterparties'].fillna(0) / 50.0
        
        merged['avg_transaction_usd_log'] = np.log1p(merged['avg_transaction_usd'].fillna(0))
        merged['max_transaction_usd_log'] = np.log1p(merged['max_transaction_usd'].fillna(0))
        
        return merged
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Adding temporal features")
        
        df['processing_date_dt'] = pd.to_datetime(df['processing_date'])
        
        df['day_of_week'] = df['processing_date_dt'].dt.dayofweek
        df['day_of_month'] = df['processing_date_dt'].dt.day
        df['month'] = df['processing_date_dt'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_month_end'] = (df['day_of_month'] >= 25).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Adding statistical features")
        
        volume = df['volume_usd'].astype(float)
        mean_vol = volume.mean()
        std_vol = volume.std()
        
        if std_vol > 0:
            df['volume_zscore'] = (volume - mean_vol) / std_vol
        else:
            df['volume_zscore'] = 0.0
        
        df['volume_percentile'] = volume.rank(pct=True)
        
        addr_stats = df.groupby('address')['volume_usd'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).add_prefix('addr_volume_')
        
        df = df.merge(addr_stats, left_on='address', right_index=True, how='left')
        df.fillna(0, inplace=True)
        
        return df
    
    def _add_cluster_features(
        self,
        alerts_df: pd.DataFrame,
        clusters_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Adding cluster features")
        
        cluster_map = {}
        for _, row in clusters_df.iterrows():
            if 'member_addresses' in row and row['member_addresses']:
                for addr in row['member_addresses']:
                    cluster_map[addr] = {
                        'cluster_id': row['cluster_id'],
                        'cluster_size': row['cluster_size'],
                        'cluster_volume': row['total_volume_usd']
                    }
        
        alerts_df['cluster_id'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_id', None)
        )
        alerts_df['cluster_size'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_size', 0)
        )
        alerts_df['cluster_volume'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_volume', 0.0)
        )
        
        alerts_df['in_cluster'] = (alerts_df['cluster_id'].notna()).astype(int)
        
        return alerts_df
    
    def _add_network_features(
        self,
        alerts_df: pd.DataFrame,
        flows_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Adding network features")
        
        inbound = flows_df.groupby('to_address').agg({
            'from_address': 'nunique',
            'amount_usd': ['sum', 'mean', 'count']
        })
        inbound.columns = ['in_degree', 'total_in', 'avg_in', 'tx_in_count']
        
        outbound = flows_df.groupby('from_address').agg({
            'to_address': 'nunique',
            'amount_usd': ['sum', 'mean', 'count']
        })
        outbound.columns = ['out_degree', 'total_out', 'avg_out', 'tx_out_count']
        
        network_features = inbound.join(outbound, how='outer').fillna(0)
        network_features['total_degree'] = (
            network_features['in_degree'] + network_features['out_degree']
        )
        
        alerts_df = alerts_df.merge(
            network_features,
            left_on='address',
            right_index=True,
            how='left'
        )
        
        alerts_df.fillna(0, inplace=True)
        
        return alerts_df
    
    def _add_label_features(
        self,
        alerts_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Adding address label features")
        
        label_map = {}
        for _, row in labels_df.iterrows():
            addr = row['address']
            if addr not in label_map:
                label_map[addr] = {
                    'risk_level': row['risk_level'],
                    'confidence_score': row['confidence_score']
                }
        
        risk_level_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        
        alerts_df['label_risk_level'] = alerts_df['address'].map(
            lambda x: risk_level_map.get(label_map.get(x, {}).get('risk_level', 'medium'), 2)
        )
        
        alerts_df['has_label'] = alerts_df['address'].map(
            lambda x: 1 if x in label_map else 0
        )
        
        alerts_df['label_confidence'] = alerts_df['address'].map(
            lambda x: label_map.get(x, {}).get('confidence_score', 0.0)
        )
        
        return alerts_df
    
    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info("Finalizing feature matrix")
        
        drop_cols = [
            'alert_id', 'address', 'processing_date', 'processing_date_dt',
            'typology_type', 'pattern_id', 'pattern_type',
            'severity', 'suspected_address_type', 'suspected_address_subtype',
            'description', 'evidence_json', 'risk_indicators',
            'label', 'ground_truth', 'cluster_id'
        ]
        
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=existing_drop_cols)
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Final feature matrix: {X.shape}")
        
        return X