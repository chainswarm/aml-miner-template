import pandas as pd
from typing import Dict
from loguru import logger


class MinerError(Exception):
    pass


class ModelNotLoadedError(MinerError):
    pass


class InvalidBatchError(MinerError):
    pass


class FeatureComputationError(MinerError):
    pass


class PredictionError(MinerError):
    pass


def validate_batch_data(batch: Dict[str, pd.DataFrame]) -> None:
    required_keys = ["alerts", "features", "clusters"]
    
    for key in required_keys:
        if key not in batch:
            raise InvalidBatchError(f"Missing required key in batch: {key}")
        
        if not isinstance(batch[key], pd.DataFrame):
            raise InvalidBatchError(f"Batch key '{key}' must be a pandas DataFrame")
        
        if batch[key].empty:
            raise InvalidBatchError(f"DataFrame '{key}' is empty")
    
    logger.debug(f"Batch validation passed: {len(batch['alerts'])} alerts, {len(batch['features'])} features, {len(batch['clusters'])} clusters")


def validate_alerts_df(df: pd.DataFrame) -> None:
    required_columns = [
        "window_days",
        "processing_date",
        "alert_id",
        "address",
        "typology_type",
        "severity",
        "alert_confidence_score",
        "description",
        "volume_usd",
        "evidence_json",
        "risk_indicators",
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidBatchError(f"Alerts DataFrame missing required columns: {missing_columns}")
    
    if df["alert_id"].duplicated().any():
        raise InvalidBatchError("Alerts DataFrame contains duplicate alert_id values")
    
    if df["alert_id"].isnull().any():
        raise InvalidBatchError("Alerts DataFrame contains null alert_id values")
    
    logger.debug(f"Alerts DataFrame validation passed: {len(df)} rows")


def validate_features_df(df: pd.DataFrame) -> None:
    required_columns = [
        "window_days",
        "processing_date",
        "address",
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidBatchError(f"Features DataFrame missing required columns: {missing_columns}")
    
    if df["address"].isnull().any():
        raise InvalidBatchError("Features DataFrame contains null address values")
    
    logger.debug(f"Features DataFrame validation passed: {len(df)} rows")


def validate_clusters_df(df: pd.DataFrame) -> None:
    required_columns = [
        "window_days",
        "processing_date",
        "cluster_id",
        "cluster_type",
        "primary_alert_id",
        "related_alert_ids",
        "addresses_involved",
        "total_alerts",
        "total_volume_usd",
    ]
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise InvalidBatchError(f"Clusters DataFrame missing required columns: {missing_columns}")
    
    if df["cluster_id"].duplicated().any():
        raise InvalidBatchError("Clusters DataFrame contains duplicate cluster_id values")
    
    if df["cluster_id"].isnull().any():
        raise InvalidBatchError("Clusters DataFrame contains null cluster_id values")
    
    logger.debug(f"Clusters DataFrame validation passed: {len(df)} rows")