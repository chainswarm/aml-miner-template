# ML Training Implementation Guide

**Date**: 2025-10-29  
**Purpose**: Detailed implementation specification for ML training pipeline  
**Status**: Implementation Ready

---

## Table of Contents

1. [Training Pipeline Specification](#training-pipeline-specification)
2. [Feature Engineering Workflow](#feature-engineering-workflow)
3. [Model Training Strategy](#model-training-strategy)
4. [Model Storage & Versioning](#model-storage--versioning)
5. [Integration with Ingestion](#integration-with-ingestion)
6. [Implementation Checklist](#implementation-checklist)

---

## Training Pipeline Specification

### Phase 1: Data Extraction

**Module**: `packages/training/feature_extraction.py`

```python
from typing import Dict, List
import pandas as pd
from clickhouse_connect.driver import Client
from loguru import logger

class FeatureExtractor:
    
    def __init__(self, client: Client):
        self.client = client
    
    def extract_training_data(
        self,
        network: str,
        start_date: str,
        end_date: str,
        window_days: int = 7
    ) -> Dict[str, pd.DataFrame]:
        """
        Extract all data needed for training from ClickHouse.
        
        Args:
            network: Network identifier (ethereum, bitcoin, etc.)
            start_date: Training period start (YYYY-MM-DD)
            end_date: Training period end (YYYY-MM-DD)
            window_days: Time window for features (default: 7)
        
        Returns:
            Dictionary with alerts, features, clusters, money_flows DataFrames
        
        Raises:
            ValueError: If no data found or required columns missing
        """
        
        logger.info(
            "Extracting training data",
            extra={
                "network": network,
                "start_date": start_date,
                "end_date": end_date,
                "window_days": window_days
            }
        )
        
        # Extract each data type
        data = {
            'alerts': self._extract_alerts(start_date, end_date, window_days),
            'features': self._extract_features(start_date, end_date, window_days),
            'clusters': self._extract_clusters(start_date, end_date, window_days),
            'money_flows': self._extract_money_flows(start_date, end_date, window_days)
        }
        
        # Validate extraction
        self._validate_extracted_data(data)
        
        logger.success(
            "Data extraction completed",
            extra={
                "alerts": len(data['alerts']),
                "features": len(data['features']),
                "clusters": len(data['clusters']),
                "money_flows": len(data['money_flows'])
            }
        )
        
        return data
    
    def _extract_alerts(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        """Extract raw alerts with all fields"""
        
        query = f"""
            SELECT
                alert_id,
                address,
                processing_date,
                window_days,
                severity,
                alert_confidence_score,
                volume_usd,
                transaction_count,
                first_seen_timestamp,
                last_seen_timestamp,
                alert_metadata
            FROM raw_alerts
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, alert_id
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            raise ValueError(
                f"No alerts found for {start_date} to {end_date}"
            )
        
        df = pd.DataFrame(
            result.result_rows,
            columns=[col[0] for col in result.column_names]
        )
        
        logger.info(f"Extracted {len(df):,} alerts")
        return df
    
    def _extract_features(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        """Extract address features"""
        
        query = f"""
            SELECT
                address,
                processing_date,
                window_days,
                total_received_usd,
                total_sent_usd,
                transaction_count,
                unique_counterparties,
                avg_transaction_usd,
                max_transaction_usd,
                is_exchange,
                is_mixer,
                risk_score,
                features_metadata
            FROM raw_features
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, address
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            raise ValueError(
                f"No features found for {start_date} to {end_date}"
            )
        
        df = pd.DataFrame(
            result.result_rows,
            columns=[col[0] for col in result.column_names]
        )
        
        logger.info(f"Extracted {len(df):,} feature records")
        return df
    
    def _extract_clusters(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        """Extract cluster information"""
        
        query = f"""
            SELECT
                cluster_id,
                processing_date,
                window_days,
                cluster_size,
                total_volume_usd,
                member_addresses,
                cluster_metadata
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
            columns=[col[0] for col in result.column_names]
        )
        
        logger.info(f"Extracted {len(df):,} clusters")
        return df
    
    def _extract_money_flows(
        self,
        start_date: str,
        end_date: str,
        window_days: int
    ) -> pd.DataFrame:
        """Extract money flow network data"""
        
        query = f"""
            SELECT
                from_address,
                to_address,
                processing_date,
                window_days,
                amount_usd,
                transaction_hash,
                timestamp
            FROM raw_money_flows
            WHERE processing_date >= '{start_date}'
              AND processing_date <= '{end_date}'
              AND window_days = {window_days}
            ORDER BY processing_date, timestamp
        """
        
        result = self.client.query(query)
        
        if not result.result_rows:
            logger.warning("No money flows found - this is optional")
            return pd.DataFrame()
        
        df = pd.DataFrame(
            result.result_rows,
            columns=[col[0] for col in result.column_names]
        )
        
        logger.info(f"Extracted {len(df):,} money flows")
        return df
    
    def _validate_extracted_data(self, data: Dict[str, pd.DataFrame]):
        """Validate extracted data completeness"""
        
        # Alerts are required
        if data['alerts'].empty:
            raise ValueError("Alerts dataframe is empty")
        
        # Features are required
        if data['features'].empty:
            raise ValueError("Features dataframe is empty")
        
        # Validate required columns exist
        required_alert_cols = ['alert_id', 'address', 'processing_date']
        missing_cols = set(required_alert_cols) - set(data['alerts'].columns)
        if missing_cols:
            raise ValueError(f"Missing required alert columns: {missing_cols}")
        
        required_feature_cols = ['address', 'processing_date']
        missing_cols = set(required_feature_cols) - set(data['features'].columns)
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        logger.info("Data validation passed")
```

---

## Feature Engineering Workflow

### Phase 2: Feature Building

**Module**: `packages/training/feature_builder.py`

```python
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from loguru import logger

class FeatureBuilder:
    
    def build_training_features(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Build complete feature matrix for training.
        
        Args:
            data: Dict with alerts, features, clusters, money_flows
        
        Returns:
            Tuple of (X: features DataFrame, y: labels Series)
        
        Raises:
            ValueError: If required data missing or invalid
        """
        
        logger.info("Building training features")
        
        # Start with alerts as base
        X = data['alerts'].copy()
        
        # Add label (for supervised learning)
        # Assuming we have a ground_truth column or derive it
        if 'label' not in X.columns and 'ground_truth' not in X.columns:
            raise ValueError(
                "No label column found in alerts. "
                "Need 'label' or 'ground_truth' for supervised learning"
            )
        
        y = X.get('label', X.get('ground_truth'))
        
        # Build feature groups
        X = self._add_alert_features(X)
        X = self._add_address_features(X, data['features'])
        X = self._add_temporal_features(X)
        X = self._add_statistical_features(X)
        
        if not data['clusters'].empty:
            X = self._add_cluster_features(X, data['clusters'])
        
        if not data['money_flows'].empty:
            X = self._add_network_features(X, data['money_flows'])
        
        # Clean up
        X = self._finalize_features(X)
        
        logger.success(
            "Feature building completed",
            extra={
                "num_samples": len(X),
                "num_features": len(X.columns),
                "positive_rate": y.mean()
            }
        )
        
        return X, y
    
    def _add_alert_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add alert-level features"""
        
        logger.info("Adding alert-level features")
        
        # Encode severity
        severity_map = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}
        df['severity_encoded'] = df['severity'].map(severity_map).fillna(2)
        
        # Log transform volume
        df['volume_usd_log'] = np.log1p(df['volume_usd'].astype(float))
        
        # Normalize transaction count
        df['transaction_count_norm'] = df['transaction_count'] / 100.0
        
        # Confidence score
        df['confidence_score'] = df['alert_confidence_score'].fillna(0.5)
        
        return df
    
    def _add_address_features(
        self,
        alerts_df: pd.DataFrame,
        features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Join and add address-level features"""
        
        logger.info("Adding address-level features")
        
        # Merge alerts with features
        merged = alerts_df.merge(
            features_df,
            on=['address', 'processing_date', 'window_days'],
            how='left',
            suffixes=('', '_feat')
        )
        
        # Feature engineering on merged data
        merged['total_volume'] = (
            merged['total_received_usd'].fillna(0) +
            merged['total_sent_usd'].fillna(0)
        )
        
        merged['volume_ratio'] = (
            merged['total_sent_usd'] /
            (merged['total_received_usd'] + 1.0)
        )
        
        # Binary flags
        merged['is_exchange_flag'] = merged['is_exchange'].fillna(False).astype(int)
        merged['is_mixer_flag'] = merged['is_mixer'].fillna(False).astype(int)
        
        # Risk score
        merged['address_risk_score'] = merged['risk_score'].fillna(0.0)
        
        return merged
    
    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        
        logger.info("Adding temporal features")
        
        # Convert to datetime
        df['processing_date_dt'] = pd.to_datetime(df['processing_date'])
        
        # Extract temporal components
        df['day_of_week'] = df['processing_date_dt'].dt.dayofweek
        df['day_of_month'] = df['processing_date_dt'].dt.day
        df['month'] = df['processing_date_dt'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Time since first seen
        if 'first_seen_timestamp' in df.columns:
            df['first_seen_dt'] = pd.to_datetime(df['first_seen_timestamp'])
            df['days_since_first_seen'] = (
                df['processing_date_dt'] - df['first_seen_dt']
            ).dt.days
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical aggregation features"""
        
        logger.info("Adding statistical features")
        
        # Z-scores for volume
        volume = df['volume_usd'].astype(float)
        mean_vol = volume.mean()
        std_vol = volume.std()
        
        if std_vol > 0:
            df['volume_zscore'] = (volume - mean_vol) / std_vol
        else:
            df['volume_zscore'] = 0.0
        
        # Percentile rank
        df['volume_percentile'] = volume.rank(pct=True)
        
        # Per-address statistics
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
        """Add cluster-based features"""
        
        logger.info("Adding cluster features")
        
        # Create address to cluster mapping
        cluster_map = {}
        for _, row in clusters_df.iterrows():
            if 'member_addresses' in row and row['member_addresses']:
                for addr in row['member_addresses']:
                    cluster_map[addr] = {
                        'cluster_id': row['cluster_id'],
                        'cluster_size': row['cluster_size'],
                        'cluster_volume': row['total_volume_usd']
                    }
        
        # Map to alerts
        alerts_df['cluster_id'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_id', None)
        )
        alerts_df['cluster_size'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_size', 0)
        )
        alerts_df['cluster_volume'] = alerts_df['address'].map(
            lambda x: cluster_map.get(x, {}).get('cluster_volume', 0.0)
        )
        
        # Binary flag for cluster membership
        alerts_df['in_cluster'] = (alerts_df['cluster_id'].notna()).astype(int)
        
        return alerts_df
    
    def _add_network_features(
        self,
        alerts_df: pd.DataFrame,
        flows_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add network graph features"""
        
        logger.info("Adding network features")
        
        # Inbound flows
        inbound = flows_df.groupby('to_address').agg({
            'from_address': 'nunique',  # in-degree
            'amount_usd': ['sum', 'mean', 'count']
        })
        inbound.columns = ['in_degree', 'total_in', 'avg_in', 'tx_in_count']
        
        # Outbound flows
        outbound = flows_df.groupby('from_address').agg({
            'to_address': 'nunique',  # out-degree
            'amount_usd': ['sum', 'mean', 'count']
        })
        outbound.columns = ['out_degree', 'total_out', 'avg_out', 'tx_out_count']
        
        # Combine
        network_features = inbound.join(outbound, how='outer').fillna(0)
        network_features['total_degree'] = (
            network_features['in_degree'] + network_features['out_degree']
        )
        
        # Merge with alerts
        alerts_df = alerts_df.merge(
            network_features,
            left_on='address',
            right_index=True,
            how='left'
        )
        
        alerts_df.fillna(0, inplace=True)
        
        return alerts_df
    
    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean up and prepare final feature matrix"""
        
        logger.info("Finalizing feature matrix")
        
        # Drop non-feature columns
        drop_cols = [
            'alert_id', 'address', 'processing_date', 'processing_date_dt',
            'first_seen_timestamp', 'last_seen_timestamp', 'first_seen_dt',
            'alert_metadata', 'features_metadata', 'cluster_metadata',
            'label', 'ground_truth'  # Target variable
        ]
        
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        X = df.drop(columns=existing_drop_cols)
        
        # Select only numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Final cleanup
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Final feature matrix: {X.shape}")
        
        return X
```

---

## Model Training Strategy

### Phase 3: Model Training

**Module**: `packages/training/model_trainer.py`

```python
import lightgbm as lgb
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from loguru import logger

class ModelTrainer:
    
    def __init__(self, model_type: str = 'alert_scorer'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Dict = None,
        cv_folds: int = 5
    ) -> Tuple[lgb.Booster, Dict]:
        """
        Train LightGBM model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            hyperparameters: Model hyperparameters (optional)
            cv_folds: Number of CV folds (default: 5)
        
        Returns:
            Tuple of (trained model, metrics dict)
        """
        
        logger.info(
            "Starting model training",
            extra={
                "model_type": self.model_type,
                "num_samples": len(X),
                "num_features": len(X.columns),
                "cv_folds": cv_folds
            }
        )
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Default hyperparameters
        if hyperparameters is None:
            hyperparameters = self._get_default_hyperparameters()
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        logger.info(f"Train set: {len(X_train):,} samples")
        logger.info(f"Test set: {len(X_test):,} samples")
        
        # Create datasets
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names
        )
        
        test_data = lgb.Dataset(
            X_test,
            label=y_test,
            reference=train_data,
            feature_name=self.feature_names
        )
        
        # Train model
        self.model = lgb.train(
            hyperparameters,
            train_data,
            valid_sets=[train_data, test_data],
            valid_names=['train', 'test'],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(10)
            ]
        )
        
        # Evaluate on test set
        test_metrics = self._evaluate_model(X_test, y_test)
        
        # Cross-validation
        if cv_folds > 1:
            cv_metrics = self._cross_validate(X_train, y_train, hyperparameters, cv_folds)
            test_metrics.update(cv_metrics)
        
        logger.success(
            "Training completed",
            extra={"metrics": test_metrics}
        )
        
        return self.model, test_metrics
    
    def _get_default_hyperparameters(self) -> Dict:
        """Get default hyperparameters based on model type"""
        
        if self.model_type == 'alert_scorer':
            return {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'verbose': -1
            }
        elif self.model_type == 'alert_ranker':
            return {
                'objective': 'lambdarank',
                'metric': 'ndcg',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': -1,
                'verbose': -1
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _evaluate_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model on test set"""
        
        logger.info("Evaluating model on test set")
        
        y_pred_proba = self.model.predict(X_test)
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        metrics = {
            'test_auc': float(roc_auc_score(y_test, y_pred_proba)),
            'test_precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'best_iteration': self.model.best_iteration,
            'num_trees': self.model.num_trees()
        }
        
        logger.info(
            "Test metrics",
            extra={
                "auc": f"{metrics['test_auc']:.4f}",
                "precision": f"{metrics['test_precision']:.4f}",
                "recall": f"{metrics['test_recall']:.4f}",
                "f1": f"{metrics['test_f1']:.4f}"
            }
        )
        
        return metrics
    
    def _cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        hyperparameters: Dict,
        n_folds: int
    ) -> Dict:
        """Perform cross-validation"""
        
        logger.info(f"Performing {n_folds}-fold cross-validation")
        
        cv_scores = []
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
            X_fold_train = X.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            fold_train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            fold_val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=fold_train_data)
            
            fold_model = lgb.train(
                hyperparameters,
                fold_train_data,
                valid_sets=[fold_val_data],
                valid_names=['validation'],
                callbacks=[
                    lgb.early_stopping(50),
                    lgb.log_evaluation(0)
                ]
            )
            
            y_val_pred = fold_model.predict(X_fold_val)
            fold_auc = roc_auc_score(y_fold_val, y_val_pred)
            cv_scores.append(fold_auc)
            
            logger.info(f"Fold {fold} AUC: {fold_auc:.4f}")
        
        cv_metrics = {
            'cv_auc_mean': float(np.mean(cv_scores)),
            'cv_auc_std': float(np.std(cv_scores)),
            'cv_scores': [float(s) for s in cv_scores]
        }
        
        logger.info(
            f"Cross-validation: {cv_metrics['cv_auc_mean']:.4f} ± {cv_metrics['cv_auc_std']:.4f}"
        )
        
        return cv_metrics
```

---

## Model Storage & Versioning

### Phase 4: Model Persistence

**Module**: `packages/training/model_storage.py`

```python
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import lightgbm as lgb
from loguru import logger
from clickhouse_connect.driver import Client

class ModelStorage:
    
    def __init__(self, output_dir: Path, client: Optional[Client] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
    
    def save_model(
        self,
        model: lgb.Booster,
        model_type: str,
        network: str,
        metrics: Dict[str, float],
        training_config: Dict[str, Any]
    ) -> Path:
        """
        Save model with complete metadata.
        
        Args:
            model: Trained LightGBM model
            model_type: Type of model (alert_scorer, alert_ranker, etc.)
            network: Network identifier
            metrics: Training metrics
            training_config: Configuration used for training
        
        Returns:
            Path to saved model file
        """
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        version = training_config.get('version', '1.0.0')
        
        # Generate filenames
        model_filename = f"{model_type}_{network}_v{version}_{timestamp}.txt"
        model_path = self.output_dir / model_filename
        
        # Save LightGBM model
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        # Create metadata
        metadata = self._create_metadata(
            model=model,
            model_type=model_type,
            network=network,
            version=version,
            timestamp=timestamp,
            metrics=metrics,
            training_config=training_config
        )
        
        # Save metadata JSON
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.success(f"Metadata saved to {metadata_path}")
        
        # Store in ClickHouse if client available
        if self.client:
            self._store_in_clickhouse(model_path, metadata)
        
        return model_path
    
    def _create_metadata(
        self,
        model: lgb.Booster,
        model_type: str,
        network: str,
        version: str,
        timestamp: str,
        metrics: Dict[str, float],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive model metadata"""
        
        return {
            'model_id': f"{model_type}_{network}_v{version}_{timestamp}",
            'model_type': model_type,
            'network': network,
            'version': version,
            'created_at': datetime.utcnow().isoformat(),
            'training_period': {
                'start_date': training_config.get('start_date'),
                'end_date': training_config.get('end_date'),
                'window_days': training_config.get('window_days', 7)
            },
            'data_stats': {
                'num_samples': training_config.get('num_samples', 0),
                'num_features': model.num_feature(),
                'positive_rate': training_config.get('positive_rate', 0.0)
            },
            'metrics': metrics,
            'hyperparameters': model.params,
            'feature_names': model.feature_name(),
            'num_trees': model.num_trees(),
            'best_iteration': model.best_iteration
        }
    
    def _store_in_clickhouse(self, model_path: Path, metadata: Dict[str, Any]):
        """Store model metadata in ClickHouse for tracking"""
        
        logger.info("Storing model metadata in ClickHouse")
        
        insert_query = """
            INSERT INTO trained_models (
                model_id,
                model_type,
                version,
                network,
                training_start_date,
                training_end_date,
                created_at,
                model_path,
                metrics_json,
                hyperparameters_json,
                feature_names,
                num_samples,
                num_features,
                test_auc,
                cv_auc_mean,
                cv_auc_std
            ) VALUES
        """
        
        values = (
            metadata['model_id'],
            metadata['model_type'],
            metadata['version'],
            metadata['network'],
            metadata['training_period']['start_date'],
            metadata['training_period']['end_date'],
            datetime.utcnow(),
            str(model_path),
            json.dumps(metadata['metrics']),
            json.dumps(metadata['hyperparameters']),
            metadata['feature_names'],
            metadata['data_stats']['num_samples'],
            metadata['data_stats']['num_features'],
            metadata['metrics'].get('test_auc', 0.0),
            metadata['metrics'].get('cv_auc_mean', 0.0),
            metadata['metrics'].get('cv_auc_std', 0.0)
        )
        
        self.client.insert(
            'trained_models',
            [values],
            column_names=[
                'model_id', 'model_type', 'version', 'network',
                'training_start_date', 'training_end_date', 'created_at',
                'model_path', 'metrics_json', 'hyperparameters_json',
                'feature_names', 'num_samples', 'num_features',
                'test_auc', 'cv_auc_mean', 'cv_auc_std'
            ]
        )
        
        logger.success("Model metadata stored in ClickHouse")
    
    def load_model(self, model_path: Path) -> lgb.Booster:
        """Load model from disk"""
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = lgb.Booster(model_file=str(model_path))
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def get_latest_model(
        self,
        model_type: str,
        network: str = None
    ) -> Path:
        """Get path to most recent model"""
        
        if network:
            pattern = f"{model_type}_{network}_v*.txt"
        else:
            pattern = f"{model_type}_*_v*.txt"
        
        models = sorted(self.output_dir.glob(pattern), reverse=True)
        
        if not models:
            raise FileNotFoundError(
                f"No models found for type={model_type}, network={network}"
            )
        
        latest = models[0]
        logger.info(f"Latest model: {latest}")
        
        return latest
```

---

## Integration with Ingestion

### Complete Training Pipeline

**Module**: `packages/training/model_training.py`

```python
from abc import ABC
from pathlib import Path
from typing import Dict, Any
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import ClientFactory, get_connection_params
from packages.training.feature_extraction import FeatureExtractor
from packages.training.feature_builder import FeatureBuilder
from packages.training.model_trainer import ModelTrainer
from packages.training.model_storage import ModelStorage

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

class ModelTraining(ABC):
    
    def __init__(
        self,
        network: str,
        start_date: str,
        end_date: str,
        client,
        model_type: str = 'alert_scorer',
        output_dir: Path = None
    ):
        self.network = network
        self.start_date = start_date
        self.end_date = end_date
        self.client = client
        self.model_type = model_type
        
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'trained_models' / network
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Main training workflow"""
        
        if terminate_event.is_set():
            logger.info("Termination requested before start")
            return
        
        logger.info(
            "Starting training workflow",
            extra={
                "network": self.network,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "model_type": self.model_type
            }
        )
        
        # Phase 1: Extract data
        logger.info("Extracting training data from ClickHouse")
        extractor = FeatureExtractor(self.client)
        data = extractor.extract_training_data(
            network=self.network,
            start_date=self.start_date,
            end_date=self.end_date
        )
        
        if terminate_event.is_set():
            logger.warning("Termination requested after extraction")
            return
        
        # Phase 2: Build features
        logger.info("Building feature matrix")
        builder = FeatureBuilder()
        X, y = builder.build_training_features(data)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after feature building")
            return
        
        # Phase 3: Train model
        logger.info("Training model")
        trainer = ModelTrainer(model_type=self.model_type)
        model, metrics = trainer.train(X, y, cv_folds=5)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after training")
            return
        
        # Phase 4: Save model
        logger.info("Saving model and metadata")
        storage = ModelStorage(self.output_dir, self.client)
        
        training_config = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'window_days': 7,
            'num_samples': len(X),
            'positive_rate': float(y.mean()),
            'version': '1.0.0'
        }
        
        model_path = storage.save_model(
            model=model,
            model_type=self.model_type,
            network=self.network,
            metrics=metrics,
            training_config=training_config
        )
        
        logger.success(
            "Training workflow completed successfully",
            extra={
                "model_path": str(model_path),
                "test_auc": metrics.get('test_auc', 0.0),
                "cv_auc_mean": metrics.get('cv_auc_mean', 0.0)
            }
        )

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--start-date', type=str, required=True)
    parser.add_argument('--end-date', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='alert_scorer',
                       choices=['alert_scorer', 'alert_ranker', 'cluster_scorer'])
    parser.add_argument('--output-dir', type=Path, default=None)
    args = parser.parse_args()
    
    service_name = f'{args.network}-{args.model_type}-training'
    setup_logger(service_name)
    load_dotenv()
    
    logger.info(
        "Initializing model training",
        extra={
            "network": args.network,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "model_type": args.model_type
        }
    )
    
    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        training = ModelTraining(
            network=args.network,
            start_date=args.start_date,
            end_date=args.end_date,
            client=client,
            model_type=args.model_type,
            output_dir=args.output_dir
        )
        
        training.run()
```

---

## Implementation Checklist

### Phase 1: Core Infrastructure ✅

- [ ] Create `packages/training/__init__.py`
- [ ] Create `packages/training/feature_extraction.py`
- [ ] Create `packages/training/feature_builder.py`
- [ ] Create `packages/training/model_trainer.py`
- [ ] Create `packages/training/model_storage.py`
- [ ] Create `packages/training/model_training.py`

### Phase 2: Database Schema ✅

- [ ] Create `packages/storage/schema/trained_models.sql`
- [ ] Update `packages/storage/__init__.py` to include new schema
- [ ] Test schema creation

### Phase 3: Testing ✅

- [ ] Test data extraction from ClickHouse
- [ ] Test feature building with sample data
- [ ] Test model training with small dataset
- [ ] Test model saving and loading
- [ ] Test metadata storage in ClickHouse

### Phase 4: Integration ✅

- [ ] Test end-to-end training workflow
- [ ] Test with multiple networks
- [ ] Test with different date ranges
- [ ] Verify logging and error handling
- [ ] Test terminate_event handling

### Phase 5: Documentation ✅

- [ ] Update README with training examples
- [ ] Create training workflow diagram
- [ ] Document hyperparameter tuning
- [ ] Add troubleshooting guide

---

## Usage Examples

### Basic Training

```bash
# Train alert scorer for ethereum
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_scorer
```

### Training All Model Types

```bash
# Alert scorer
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_scorer

# Alert ranker
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type alert_ranker

# Cluster scorer
python -m packages.training.model_training \
    --network ethereum \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type cluster_scorer
```

### Multi-Network Training

```bash
# Train for multiple networks
for network in ethereum bitcoin polygon; do
    python -m packages.training.model_training \
        --network $network \
        --start-date 2024-01-01 \
        --end-date 2024-03-31 \
        --model-type alert_scorer
done
```

---

## Success Criteria

✅ **Data Extraction**: Successfully queries ClickHouse and loads data  
✅ **Feature Building**: Creates complete feature matrix  
✅ **Model Training**: Trains LightGBM model with CV  
✅ **Model Storage**: Saves model with metadata  
✅ **Integration**: Works seamlessly with ingestion pipeline  
✅ **Error Handling**: Fails fast with clear errors  
✅ **Logging**: Rich, contextual logging throughout  
✅ **Versioning**: Proper model versioning and tracking  

---

## Next Steps

1. Implement core training classes
2. Create database schema for model tracking
3. Test with real ingested data
4. Optimize hyperparameters
5. Create automated training workflows
6. Deploy trained models for inference
