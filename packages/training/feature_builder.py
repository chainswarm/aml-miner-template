import pandas as pd
import numpy as np
from decimal import Decimal
from typing import Dict, Tuple, List
from loguru import logger


class FeatureBuilder:
    
    def _log_feature_statistics(self, df: pd.DataFrame, stage: str):
        
        logger.info(f"Feature statistics at stage: {stage}")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:10]:
            values = df[col].dropna()
            if len(values) > 0:
                logger.info(
                    f"  {col}",
                    extra={
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "std": float(values.std()) if len(values) > 1 else 0.0,
                        "unique": int(values.nunique())
                    }
                )
        
        if len(numeric_cols) > 10:
            logger.info(f"  ... and {len(numeric_cols) - 10} more features")
    
    def build_inference_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        
        logger.info("Building inference features")
        
        X = data['alerts'].copy()
        
        X = self._add_alert_features(X)
        self._log_feature_statistics(X, "after_alert_features")
        
        X = self._add_address_features(X, data['features'])
        self._log_feature_statistics(X, "after_address_features")
        
        X = self._add_temporal_features(X)
        self._log_feature_statistics(X, "after_temporal_features")
        
        X = self._add_statistical_features(X)
        self._log_feature_statistics(X, "after_statistical_features")
        
        if not data['clusters'].empty:
            X = self._add_cluster_features(X, data['clusters'])
            self._log_feature_statistics(X, "after_cluster_features")
        
        if not data['money_flows'].empty:
            X = self._add_network_features(X, data['money_flows'])
            self._log_feature_statistics(X, "after_network_features")
        
        if not data['address_labels'].empty:
            X = self._add_label_features(X, data['address_labels'])
            self._log_feature_statistics(X, "after_label_features")
        
        X = self._finalize_features(X)
        self._log_feature_statistics(X, "final_features")
        
        logger.success(
            "Inference feature building completed",
            extra={
                "num_samples": len(X),
                "num_features": len(X.columns)
            }
        )
        
        return X
    
    def build_training_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        
        logger.info("Building training features")
        
        alerts_with_labels = self._derive_labels_from_address_labels(
            data['alerts'],
            data['address_labels']
        )
        
        labeled_alerts = alerts_with_labels[
            alerts_with_labels['label'].notna()
        ].copy()
        
        if len(labeled_alerts) == 0:
            raise ValueError(
                "No labeled alerts found. "
                "address_labels table must contain labels for alert addresses"
            )
        
        y = labeled_alerts['label']
        
        X = labeled_alerts.copy()
        
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
        
        logger.success(
            "Feature building completed",
            extra={
                "num_samples": len(X),
                "num_features": len(X.columns),
                "positive_rate": float(y.mean())
            }
        )
        
        return X, y
    
    def build_cluster_training_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
        
        logger.info("Building cluster training features")
        
        alerts_with_labels = self._derive_labels_from_address_labels(
            data['alerts'],
            data['address_labels']
        )
        
        labeled_alerts = alerts_with_labels[
            alerts_with_labels['label'].notna()
        ].copy()
        
        if len(labeled_alerts) == 0:
            raise ValueError(
                "No labeled alerts found. "
                "address_labels table must contain labels for alert addresses"
            )
        
        data_with_labels = data.copy()
        data_with_labels['alerts'] = labeled_alerts
        
        cluster_labels_df = self._derive_cluster_labels(
            data['clusters'],
            labeled_alerts
        )
        
        if len(cluster_labels_df) == 0:
            logger.warning(
                "No labeled clusters found. "
                "Creating synthetic cluster features from labeled alerts"
            )
            
            synthetic_cluster_features = self._create_synthetic_cluster_features(
                data_with_labels,
                labeled_alerts
            )
            
            y = labeled_alerts['label'].reset_index(drop=True)
            
            logger.success(
                "Synthetic cluster training features created",
                extra={
                    "num_clusters": len(synthetic_cluster_features),
                    "num_features": len(synthetic_cluster_features.columns),
                    "positive_rate": float(y.mean()) if len(y) > 0 else 0.0
                }
            )
            
            return synthetic_cluster_features, y, labeled_alerts[['alert_id', 'label', 'label_confidence']].copy()
        
        all_cluster_features = self.build_cluster_features(data_with_labels)
        
        clusters_df = data['clusters'].copy()
        clusters_df['cluster_index'] = range(len(clusters_df))
        cluster_id_to_index = dict(zip(clusters_df['cluster_id'], clusters_df['cluster_index']))
        
        labeled_indices = [
            cluster_id_to_index[cid]
            for cid in cluster_labels_df['cluster_id']
            if cid in cluster_id_to_index
        ]
        
        X = all_cluster_features.iloc[labeled_indices].reset_index(drop=True)
        y = cluster_labels_df['label'].reset_index(drop=True)
        
        logger.success(
            "Cluster training feature building completed",
            extra={
                "num_clusters": len(X),
                "num_features": len(X.columns),
                "positive_rate": float(y.mean()) if len(y) > 0 else 0.0
            }
        )
        
        return X, y, cluster_labels_df
    
    def _derive_cluster_labels(
        self,
        clusters_df: pd.DataFrame,
        labeled_alerts: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Deriving labels for clusters from labeled alerts")
        
        cluster_labels = []
        
        for _, cluster_row in clusters_df.iterrows():
            cluster_id = cluster_row['cluster_id']
            
            alert_ids = []
            if 'related_alert_ids' in cluster_row and cluster_row['related_alert_ids'] is not None:
                if isinstance(cluster_row['related_alert_ids'], list) and len(cluster_row['related_alert_ids']) > 0:
                    alert_ids = cluster_row['related_alert_ids']
            
            if not alert_ids and 'primary_alert_id' in cluster_row and cluster_row['primary_alert_id']:
                alert_ids = [cluster_row['primary_alert_id']]
            
            cluster_alert_labels = labeled_alerts[
                labeled_alerts['alert_id'].isin(alert_ids)
            ]['label']
            
            if len(cluster_alert_labels) > 0:
                cluster_label = int(cluster_alert_labels.max())
                cluster_confidence = float(cluster_alert_labels.mean())
            else:
                cluster_label = None
                cluster_confidence = None
            
            cluster_labels.append({
                'cluster_id': cluster_id,
                'label': cluster_label,
                'label_confidence': cluster_confidence
            })
        
        labels_df = pd.DataFrame(cluster_labels)
        labels_df = labels_df[labels_df['label'].notna()]
        
        num_labeled = len(labels_df)
        num_positive = (labels_df['label'] == 1).sum()
        num_negative = (labels_df['label'] == 0).sum()
        
        logger.info(
            f"Labeled {num_labeled}/{len(clusters_df)} clusters: "
            f"{num_positive} positive, {num_negative} negative"
        )
        
        return labels_df
    
    def _create_synthetic_cluster_features(
        self,
        data: Dict[str, pd.DataFrame],
        labeled_alerts: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Creating synthetic cluster features from labeled alerts")
        
        alert_features = self.build_inference_features(data)
        
        alerts_df = data['alerts'].copy()
        alerts_df_indexed = alerts_df.set_index('alert_id')
        
        alert_features_with_id = alert_features.copy()
        alert_features_with_id['alert_id'] = alerts_df['alert_id'].values
        alert_features_indexed = alert_features_with_id.set_index('alert_id')
        
        cluster_features_list = []
        
        for _, alert_row in labeled_alerts.iterrows():
            alert_id = alert_row['alert_id']
            
            if alert_id not in alert_features_indexed.index:
                continue
            
            alert_feature_row = alert_features_indexed.loc[alert_id]
            alert_data = alerts_df_indexed.loc[alert_id]
            
            features = {
                'cluster_size': 1.0,
                'cluster_total_volume': float(alert_data.get('volume_usd', 0)),
                'cluster_avg_confidence': float(alert_data.get('alert_confidence_score', 0.5)),
            }
            
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            severity = alert_data.get('severity', 'medium')
            features['cluster_max_severity'] = float(severity_map.get(severity, 2))
            
            cluster_type_map = {
                'structural': 1, 'temporal': 2, 'behavioral': 3,
                'mixed': 4, 'unknown': 0
            }
            features['cluster_type_encoded'] = 0.0
            
            features['alert_count'] = 1.0
            features['alert_volume_sum'] = float(alert_data.get('volume_usd', 0))
            features['alert_volume_mean'] = float(alert_data.get('volume_usd', 0))
            features['alert_volume_std'] = 0.0
            features['alert_volume_max'] = float(alert_data.get('volume_usd', 0))
            features['alert_volume_min'] = float(alert_data.get('volume_usd', 0))
            
            if 'alert_confidence_score' in alert_data:
                conf = float(alert_data['alert_confidence_score'])
                features['alert_confidence_mean'] = conf
                features['alert_confidence_std'] = 0.0
            
            if 'severity' in alert_data:
                sev_val = float(severity_map.get(alert_data['severity'], 2))
                features['alert_severity_mean'] = sev_val
                features['alert_high_severity_count'] = 1.0 if sev_val >= 3 else 0.0
                features['alert_critical_severity_count'] = 1.0 if sev_val == 4 else 0.0
            
            for col in alert_feature_row.index:
                if pd.api.types.is_numeric_dtype(type(alert_feature_row[col])):
                    val = float(alert_feature_row[col])
                    features[f'agg_{col}_mean'] = val
                    features[f'agg_{col}_std'] = 0.0
                    features[f'agg_{col}_max'] = val
                    features[f'agg_{col}_min'] = val
            
            features['unique_address_count'] = 1.0
            
            if not data['features'].empty:
                addr = alert_data.get('address')
                if addr:
                    addr_features = data['features'][data['features']['address'] == addr]
                    if len(addr_features) > 0:
                        addr_row = addr_features.iloc[0]
                        if 'behavioral_anomaly_score' in addr_row:
                            features['address_anomaly_behavioral_mean'] = float(addr_row['behavioral_anomaly_score'])
                        if 'graph_anomaly_score' in addr_row:
                            features['address_anomaly_graph_mean'] = float(addr_row['graph_anomaly_score'])
                        if 'is_exchange_like' in addr_row:
                            features['address_exchange_count'] = 1.0 if addr_row['is_exchange_like'] else 0.0
                        if 'is_mixer_like' in addr_row:
                            features['address_mixer_count'] = 1.0 if addr_row['is_mixer_like'] else 0.0
            
            cluster_features_list.append(features)
        
        result = pd.DataFrame(cluster_features_list)
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Created {len(result)} synthetic cluster feature rows")
        
        return result
    
    def build_cluster_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        
        logger.info("Building cluster features")
        
        if data['clusters'].empty:
            raise ValueError("No clusters to build features for")
        
        alert_features = self.build_inference_features(data)
        alerts_df = data['alerts'].copy()
        
        cluster_to_alerts = self._build_cluster_alert_map(
            data['clusters'],
            alerts_df
        )
        
        cluster_features = self._aggregate_cluster_features(
            data['clusters'],
            cluster_to_alerts,
            alert_features,
            alerts_df,
            data
        )
        
        logger.success(
            "Cluster feature building completed",
            extra={
                "num_clusters": len(cluster_features),
                "num_features": len(cluster_features.columns)
            }
        )
        
        return cluster_features
    
    def _build_cluster_alert_map(
        self,
        clusters_df: pd.DataFrame,
        alerts_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        
        cluster_to_alerts = {}
        
        for _, row in clusters_df.iterrows():
            cluster_id = row['cluster_id']
            alert_ids = []
            
            if 'related_alert_ids' in row and row['related_alert_ids'] is not None:
                if isinstance(row['related_alert_ids'], list) and len(row['related_alert_ids']) > 0:
                    alert_ids = row['related_alert_ids']
            
            if not alert_ids and 'primary_alert_id' in row and row['primary_alert_id']:
                alert_ids = [row['primary_alert_id']]
            
            if not alert_ids and 'addresses_involved' in row and row['addresses_involved']:
                if isinstance(row['addresses_involved'], list):
                    matching_alerts = alerts_df[
                        alerts_df['address'].isin(row['addresses_involved'])
                    ]['alert_id'].tolist()
                    if matching_alerts:
                        alert_ids = matching_alerts
            
            if alert_ids:
                cluster_to_alerts[cluster_id] = alert_ids
        
        logger.info(f"Mapped {len(cluster_to_alerts)} clusters to their alert IDs")
        
        return cluster_to_alerts
    
    def _aggregate_cluster_features(
        self,
        clusters_df: pd.DataFrame,
        cluster_to_alerts: Dict[str, List[str]],
        alert_features: pd.DataFrame,
        alerts_df: pd.DataFrame,
        data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        
        cluster_features_list = []
        
        alerts_df_indexed = alerts_df.set_index('alert_id')
        alert_features_with_id = alert_features.copy()
        alert_features_with_id['alert_id'] = alerts_df['alert_id'].values
        alert_features_indexed = alert_features_with_id.set_index('alert_id')
        
        for _, cluster_row in clusters_df.iterrows():
            cluster_id = cluster_row['cluster_id']
            
            cluster_alert_ids = cluster_to_alerts.get(cluster_id, [])
            if not cluster_alert_ids:
                continue
            
            cluster_alerts = alerts_df_indexed.loc[
                alerts_df_indexed.index.isin(cluster_alert_ids)
            ]
            cluster_alert_features = alert_features_indexed.loc[
                alert_features_indexed.index.isin(cluster_alert_ids)
            ]
            
            features = {
                'cluster_size': float(cluster_row.get('total_alerts', len(cluster_alert_ids))),
                'cluster_total_volume': float(cluster_row.get('total_volume_usd', 0)),
                'cluster_avg_confidence': float(cluster_row.get('confidence_avg', 0.5)),
            }
            
            severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
            max_severity = cluster_row.get('severity_max', 'medium')
            features['cluster_max_severity'] = float(severity_map.get(max_severity, 2))
            
            cluster_type_map = {
                'structural': 1, 'temporal': 2, 'behavioral': 3, 
                'mixed': 4, 'unknown': 0
            }
            cluster_type = cluster_row.get('cluster_type', 'unknown')
            features['cluster_type_encoded'] = float(cluster_type_map.get(cluster_type, 0))
            
            if len(cluster_alerts) > 0:
                features['alert_count'] = float(len(cluster_alerts))
                
                if 'volume_usd' in cluster_alerts.columns:
                    volumes = cluster_alerts['volume_usd'].astype(float)
                    features['alert_volume_sum'] = float(volumes.sum())
                    features['alert_volume_mean'] = float(volumes.mean())
                    features['alert_volume_std'] = float(volumes.std() if len(volumes) > 1 else 0)
                    features['alert_volume_max'] = float(volumes.max())
                    features['alert_volume_min'] = float(volumes.min())
                
                if 'alert_confidence_score' in cluster_alerts.columns:
                    confidences = cluster_alerts['alert_confidence_score'].fillna(0.5)
                    features['alert_confidence_mean'] = float(confidences.mean())
                    features['alert_confidence_std'] = float(confidences.std() if len(confidences) > 1 else 0)
                
                if 'severity' in cluster_alerts.columns:
                    severities = cluster_alerts['severity'].map(severity_map).fillna(2)
                    features['alert_severity_mean'] = float(severities.mean())
                    features['alert_high_severity_count'] = float((severities >= 3).sum())
                    features['alert_critical_severity_count'] = float((severities == 4).sum())
            
            if len(cluster_alert_features) > 0:
                numeric_features = cluster_alert_features.select_dtypes(include=[np.number])
                
                for col in numeric_features.columns:
                    if col != 'alert_id':
                        values = numeric_features[col]
                        features[f'agg_{col}_mean'] = float(values.mean())
                        features[f'agg_{col}_std'] = float(values.std() if len(values) > 1 else 0)
                        features[f'agg_{col}_max'] = float(values.max())
                        features[f'agg_{col}_min'] = float(values.min())
            
            if 'addresses_involved' in cluster_row and cluster_row['addresses_involved']:
                features['unique_address_count'] = float(len(cluster_row['addresses_involved']))
                
                if not data['features'].empty:
                    addresses = cluster_row['addresses_involved']
                    cluster_address_features = data['features'][
                        data['features']['address'].isin(addresses)
                    ]
                    
                    if len(cluster_address_features) > 0:
                        if 'behavioral_anomaly_score' in cluster_address_features.columns:
                            features['address_anomaly_behavioral_mean'] = float(
                                cluster_address_features['behavioral_anomaly_score'].mean()
                            )
                        
                        if 'graph_anomaly_score' in cluster_address_features.columns:
                            features['address_anomaly_graph_mean'] = float(
                                cluster_address_features['graph_anomaly_score'].mean()
                            )
                        
                        if 'is_exchange_like' in cluster_address_features.columns:
                            features['address_exchange_count'] = float(
                                cluster_address_features['is_exchange_like'].sum()
                            )
                        
                        if 'is_mixer_like' in cluster_address_features.columns:
                            features['address_mixer_count'] = float(
                                cluster_address_features['is_mixer_like'].sum()
                            )
                        
                        if 'degree_total' in cluster_address_features.columns:
                            features['address_degree_total_mean'] = float(
                                cluster_address_features['degree_total'].mean()
                            )
                            features['address_degree_total_sum'] = float(
                                cluster_address_features['degree_total'].sum()
                            )
            
            cluster_features_list.append(features)
        
        result = pd.DataFrame(cluster_features_list)
        result = result.fillna(0)
        result = result.replace([np.inf, -np.inf], 0)
        
        return result
    
    
    def _derive_labels_from_address_labels(
        self,
        alerts_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Deriving labels from address_labels table")
        
        if labels_df.empty:
            raise ValueError("address_labels table is empty")
        
        label_map = {}
        confidence_map = {}
        
        for _, row in labels_df.iterrows():
            addr = row['address']
            risk = row['risk_level'].lower()
            confidence = row.get('confidence_score', 1.0)
            
            if risk in ['high', 'critical']:
                label_map[addr] = 1
                confidence_map[addr] = confidence
            elif risk in ['low', 'medium']:
                label_map[addr] = 0
                confidence_map[addr] = confidence
        
        alerts_df['label'] = alerts_df['address'].map(label_map)
        alerts_df['label_confidence'] = alerts_df['address'].map(confidence_map)
        alerts_df['label_source'] = alerts_df['address'].map(
            lambda x: 'address_labels' if x in label_map else None
        )
        
        num_labeled = alerts_df['label'].notna().sum()
        num_positive = (alerts_df['label'] == 1).sum()
        num_negative = (alerts_df['label'] == 0).sum()
        
        logger.info(
            f"Labeled {num_labeled}/{len(alerts_df)} alerts: "
            f"{num_positive} positive, {num_negative} negative"
        )
        
        return alerts_df
    
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
            merged['total_in_usd'].fillna(Decimal('0')) +
            merged['total_out_usd'].fillna(Decimal('0'))
        ).astype(float)
        
        merged['volume_ratio'] = (
            merged['total_out_usd'].astype(float) /
            (merged['total_in_usd'].astype(float) + 1.0)
        )
        
        merged['is_exchange_flag'] = merged['is_exchange_like'].fillna(False).astype(int)
        merged['is_mixer_flag'] = merged['is_mixer_like'].fillna(False).astype(int)
        
        merged['behavioral_anomaly'] = merged['behavioral_anomaly_score'].fillna(0.0)
        merged['graph_anomaly'] = merged['graph_anomaly_score'].fillna(0.0)
        merged['global_anomaly'] = merged['global_anomaly_score'].fillna(0.0)
        
        merged['tx_count_norm'] = merged['tx_total_count'].fillna(0) / 100.0
        merged['unique_counterparties_norm'] = merged['unique_counterparties'].fillna(0) / 50.0
        
        merged['avg_tx_in_log'] = np.log1p(merged['avg_tx_in_usd'].fillna(Decimal('0')).astype(float))
        merged['avg_tx_out_log'] = np.log1p(merged['avg_tx_out_usd'].fillna(Decimal('0')).astype(float))
        merged['max_tx_log'] = np.log1p(merged['max_tx_usd'].fillna(Decimal('0')).astype(float))
        
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
        
        volume_series = df['volume_usd'].astype(float)
        addr_stats = df.groupby('address').apply(
            lambda x: pd.Series({
                'addr_volume_mean': x['volume_usd'].astype(float).mean(),
                'addr_volume_std': x['volume_usd'].astype(float).std(),
                'addr_volume_min': x['volume_usd'].astype(float).min(),
                'addr_volume_max': x['volume_usd'].astype(float).max(),
                'addr_volume_count': len(x)
            })
        )
        
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
            if 'addresses_involved' in row and row['addresses_involved']:
                for addr in row['addresses_involved']:
                    cluster_map[addr] = {
                        'cluster_id': row['cluster_id'],
                        'cluster_size': row['total_alerts'],
                        'cluster_volume': row['total_volume_usd']
                    }
        
        alerts_df['cluster_id'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_id', None)
        )
        alerts_df['cluster_size'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_size', 0)
        )
        alerts_df['cluster_volume'] = alerts_df['address'].map(
            lambda x: float(cluster_map.get(x, {}).get('cluster_volume', Decimal('0')))
        )
        
        alerts_df['in_cluster'] = (alerts_df['cluster_id'].notna()).astype(int)
        
        return alerts_df
    
    def _add_network_features(
        self,
        alerts_df: pd.DataFrame,
        flows_df: pd.DataFrame
    ) -> pd.DataFrame:
        
        logger.info("Adding network features")
        
        flows_df['amount_usd_sum_float'] = flows_df['amount_usd_sum'].astype(float)
        
        inbound = flows_df.groupby('to_address').agg({
            'from_address': 'nunique',
            'amount_usd_sum_float': ['sum', 'mean'],
            'tx_count': 'sum'
        })
        inbound.columns = ['in_degree', 'total_in', 'avg_in', 'tx_in_count']
        
        outbound = flows_df.groupby('from_address').agg({
            'to_address': 'nunique',
            'amount_usd_sum_float': ['sum', 'mean'],
            'tx_count': 'sum'
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
        
        if 'label_confidence' not in alerts_df.columns:
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
            'label', 'ground_truth', 'cluster_id', 'label_source'
        ]
        
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=existing_drop_cols)
        
        for col in X.columns:
            if X[col].dtype == object:
                try:
                    X[col] = X[col].astype(float)
                except (ValueError, TypeError):
                    pass
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        canonical_order = [
            'window_days', 'alert_confidence_score', 'volume_usd', 'label_confidence',
            'severity_encoded', 'volume_usd_log', 'confidence_score', 'address_type_encoded',
            'total_in_usd', 'total_out_usd', 'tx_total_count', 'unique_counterparties',
            'avg_tx_in_usd', 'avg_tx_out_usd', 'max_tx_usd', 'behavioral_anomaly_score',
            'graph_anomaly_score', 'global_anomaly_score', 'total_volume', 'volume_ratio',
            'is_exchange_flag', 'is_mixer_flag', 'behavioral_anomaly', 'graph_anomaly',
            'global_anomaly', 'tx_count_norm', 'unique_counterparties_norm',
            'avg_tx_in_log', 'avg_tx_out_log', 'max_tx_log', 'day_of_week',
            'day_of_month', 'month', 'is_weekend', 'is_month_end', 'volume_zscore',
            'volume_percentile', 'addr_volume_mean', 'addr_volume_std',
            'addr_volume_min', 'addr_volume_max', 'addr_volume_count',
            'cluster_size', 'cluster_volume', 'in_cluster', 'in_degree',
            'total_in', 'avg_in', 'tx_in_count', 'out_degree', 'total_out',
            'avg_out', 'tx_out_count', 'total_degree', 'label_risk_level', 'has_label'
        ]
        
        for feature in canonical_order:
            if feature not in X.columns:
                X[feature] = 0.0
        
        unexpected = set(X.columns) - set(canonical_order)
        if unexpected:
            raise ValueError(
                f"Unexpected features: {sorted(unexpected)}. "
                f"Update canonical_order in _finalize_features()"
            )
        
        X = X[canonical_order]
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Final feature matrix: {X.shape}")
        
        return X