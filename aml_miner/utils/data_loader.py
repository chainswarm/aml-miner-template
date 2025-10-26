import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from loguru import logger
from aml_miner.utils.validators import (
    InvalidBatchError,
    validate_alerts_df,
    validate_features_df,
    validate_clusters_df,
)


class BatchDataLoader:
    
    def load_batch(self, batch_dir: Path) -> Dict[str, pd.DataFrame]:
        batch_dir = Path(batch_dir)
        
        if not batch_dir.exists():
            raise InvalidBatchError(f"Batch directory does not exist: {batch_dir}")
        
        if not batch_dir.is_dir():
            raise InvalidBatchError(f"Batch path is not a directory: {batch_dir}")
        
        logger.info(f"Loading batch from {batch_dir}")
        
        alerts_df = self._load_parquet_file(batch_dir / "alerts.parquet", "alerts")
        validate_alerts_df(alerts_df)
        
        features_df = self._load_parquet_file(batch_dir / "features.parquet", "features")
        validate_features_df(features_df)
        
        clusters_df = self._load_parquet_file(batch_dir / "clusters.parquet", "clusters")
        validate_clusters_df(clusters_df)
        
        money_flows_df = None
        money_flows_path = batch_dir / "money_flows.parquet"
        if money_flows_path.exists():
            logger.info(f"Loading optional money_flows.parquet")
            money_flows_df = pd.read_parquet(money_flows_path, engine="pyarrow")
            logger.info(f"Loaded {len(money_flows_df)} money flows")
        else:
            logger.debug("money_flows.parquet not found, skipping")
        
        logger.info(
            f"Batch loaded successfully: {len(alerts_df)} alerts, "
            f"{len(features_df)} features, {len(clusters_df)} clusters"
        )
        
        return {
            "alerts": alerts_df,
            "features": features_df,
            "clusters": clusters_df,
            "money_flows": money_flows_df,
        }
    
    def _load_parquet_file(self, file_path: Path, name: str) -> pd.DataFrame:
        if not file_path.exists():
            raise InvalidBatchError(f"Required file not found: {file_path}")
        
        logger.debug(f"Loading {name} from {file_path}")
        df = pd.read_parquet(file_path, engine="pyarrow")
        logger.debug(f"Loaded {len(df)} rows from {name}")
        
        return df