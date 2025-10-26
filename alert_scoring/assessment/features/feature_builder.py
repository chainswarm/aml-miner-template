import pandas as pd
import numpy as np
from typing import Optional
from loguru import logger


class FeatureBuilder:
    
    def build_alert_features(self, alerts_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building alert features for {len(alerts_df)} alerts")
        
        alert_features = alerts_df.merge(
            features_df,
            left_on='address',
            right_on='address',
            how='left',
            suffixes=('_alert', '_feature')
        )
        
        alert_features['alert_confidence_score'] = alerts_df['alert_confidence_score']
        alert_features['volume_usd'] = alerts_df['volume_usd'].astype(float)
        alert_features['severity_encoded'] = alerts_df['severity'].map({
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4
        }).fillna(2)
        
        logger.info(f"Built {len(alert_features.columns)} alert-level features")
        return alert_features
    
    def build_network_features(self, money_flows_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building network features from {len(money_flows_df)} money flows")
        
        in_flows = money_flows_df.groupby('dest_address').agg({
            'source_address': 'nunique',
            'amount_usd': ['sum', 'mean', 'std', 'count']
        })
        in_flows.columns = ['degree_in', 'total_in_usd', 'avg_tx_in_usd', 'std_tx_in_usd', 'tx_in_count']
        
        out_flows = money_flows_df.groupby('source_address').agg({
            'dest_address': 'nunique',
            'amount_usd': ['sum', 'mean', 'std', 'count']
        })
        out_flows.columns = ['degree_out', 'total_out_usd', 'avg_tx_out_usd', 'std_tx_out_usd', 'tx_out_count']
        
        network_features = in_flows.join(out_flows, how='outer').fillna(0)
        network_features['degree_total'] = network_features['degree_in'] + network_features['degree_out']
        network_features['total_volume_usd'] = network_features['total_in_usd'] + network_features['total_out_usd']
        network_features['net_flow_usd'] = network_features['total_in_usd'] - network_features['total_out_usd']
        network_features['tx_total_count'] = network_features['tx_in_count'] + network_features['tx_out_count']
        
        network_features.reset_index(inplace=True)
        network_features.rename(columns={'index': 'address'}, inplace=True)
        
        logger.info(f"Built {len(network_features.columns)} network features for {len(network_features)} addresses")
        return network_features
    
    def build_cluster_features(self, clusters_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building cluster features from {len(clusters_df)} clusters")
        
        cluster_features = clusters_df.copy()
        
        cluster_features['cluster_size'] = cluster_features['total_alerts']
        cluster_features['cluster_volume_usd'] = cluster_features['total_volume_usd'].astype(float)
        cluster_features['cluster_severity_encoded'] = cluster_features['severity_max'].map({
            'low': 1, 'medium': 2, 'high': 3, 'critical': 4
        }).fillna(2)
        cluster_features['cluster_confidence'] = cluster_features['confidence_avg']
        cluster_features['cluster_time_span'] = (
            cluster_features['latest_alert_timestamp'] - cluster_features['earliest_alert_timestamp']
        )
        
        cluster_lookup = {}
        for _, row in cluster_features.iterrows():
            for alert_id in row['related_alert_ids']:
                cluster_lookup[alert_id] = {
                    'cluster_id': row['cluster_id'],
                    'cluster_size': row['cluster_size'],
                    'cluster_volume_usd': row['cluster_volume_usd'],
                    'cluster_severity_encoded': row['cluster_severity_encoded'],
                    'cluster_confidence': row['cluster_confidence'],
                    'cluster_time_span': row['cluster_time_span']
                }
        
        logger.info(f"Built cluster features for {len(cluster_lookup)} alerts")
        return pd.DataFrame.from_dict(cluster_lookup, orient='index')
    
    def build_temporal_features(self, alerts_df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Building temporal features for {len(alerts_df)} alerts")
        
        temporal_features = pd.DataFrame(index=alerts_df.index)
        
        if 'processing_date' in alerts_df.columns:
            processing_dates = pd.to_datetime(alerts_df['processing_date'])
            temporal_features['hour'] = processing_dates.dt.hour
            temporal_features['day_of_week'] = processing_dates.dt.dayofweek
            temporal_features['is_weekend'] = (processing_dates.dt.dayofweek >= 5).astype(int)
            temporal_features['is_business_hours'] = (
                (processing_dates.dt.hour >= 9) & (processing_dates.dt.hour <= 17)
            ).astype(int)
            temporal_features['day_of_month'] = processing_dates.dt.day
            temporal_features['month'] = processing_dates.dt.month
        
        temporal_features['alert_id'] = alerts_df['alert_id'].values
        
        alert_counts = alerts_df.groupby('address').size().reset_index(name='alert_frequency')
        temporal_features = temporal_features.merge(
            alert_counts,
            left_on=alerts_df['address'].values,
            right_on='address',
            how='left'
        ).fillna(0)
        
        logger.info(f"Built {len(temporal_features.columns)} temporal features")
        return temporal_features
    
    def build_statistical_features(self, alerts_df: pd.DataFrame, window: str = '7d') -> pd.DataFrame:
        logger.info(f"Building statistical features for {len(alerts_df)} alerts with window={window}")
        
        statistical_features = pd.DataFrame(index=alerts_df.index)
        
        if 'volume_usd' in alerts_df.columns:
            volume_series = alerts_df['volume_usd'].astype(float)
            
            statistical_features['volume_log'] = np.log1p(volume_series)
            
            mean_volume = volume_series.mean()
            std_volume = volume_series.std()
            if std_volume > 0:
                statistical_features['volume_z_score'] = (volume_series - mean_volume) / std_volume
            else:
                statistical_features['volume_z_score'] = 0
            
            statistical_features['volume_percentile'] = volume_series.rank(pct=True)
            
            if 'alert_confidence_score' in alerts_df.columns:
                confidence_series = alerts_df['alert_confidence_score']
                mean_conf = confidence_series.mean()
                std_conf = confidence_series.std()
                if std_conf > 0:
                    statistical_features['confidence_z_score'] = (confidence_series - mean_conf) / std_conf
                else:
                    statistical_features['confidence_z_score'] = 0
                
                statistical_features['confidence_percentile'] = confidence_series.rank(pct=True)
        
        by_address = alerts_df.groupby('address')['volume_usd'].agg(['mean', 'std', 'min', 'max']).fillna(0)
        by_address.columns = ['addr_volume_mean', 'addr_volume_std', 'addr_volume_min', 'addr_volume_max']
        
        statistical_features = statistical_features.join(
            by_address,
            on=alerts_df['address'].values,
            how='left'
        ).fillna(0)
        
        logger.info(f"Built {len(statistical_features.columns)} statistical features")
        return statistical_features
    
    def build_all_features(
        self,
        alerts_df: pd.DataFrame,
        features_df: pd.DataFrame,
        clusters_df: pd.DataFrame,
        money_flows_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        logger.info("Building complete feature matrix")
        
        alert_features = self.build_alert_features(alerts_df, features_df)
        
        temporal_features = self.build_temporal_features(alerts_df)
        alert_features = alert_features.join(
            temporal_features.set_index('alert_id'),
            on='alert_id',
            how='left',
            rsuffix='_temporal'
        )
        
        statistical_features = self.build_statistical_features(alerts_df)
        alert_features = pd.concat([alert_features, statistical_features], axis=1)
        
        if len(clusters_df) > 0:
            cluster_features = self.build_cluster_features(clusters_df)
            alert_features = alert_features.join(
                cluster_features,
                on='alert_id',
                how='left',
                rsuffix='_cluster'
            )
        
        if money_flows_df is not None and len(money_flows_df) > 0:
            network_features = self.build_network_features(money_flows_df)
            alert_features = alert_features.merge(
                network_features,
                left_on='address',
                right_on='address',
                how='left',
                suffixes=('', '_network')
            )
        
        numeric_cols = alert_features.select_dtypes(include=[np.number]).columns
        alert_features[numeric_cols] = alert_features[numeric_cols].fillna(0)
        
        categorical_cols = alert_features.select_dtypes(include=['object']).columns
        alert_features[categorical_cols] = alert_features[categorical_cols].fillna('')
        
        logger.info(f"Complete feature matrix built: {alert_features.shape[0]} rows × {alert_features.shape[1]} columns")
        return alert_features