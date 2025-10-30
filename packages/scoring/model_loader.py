import json
from pathlib import Path
from typing import Union, Dict, Any
import xgboost as xgb
import lightgbm as lgb
from loguru import logger


class ModelLoader:
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}
    
    def load_latest_model(
        self,
        network: str,
        model_type: str
    ) -> Union[xgb.XGBClassifier, lgb.Booster]:
        
        cache_key = f"{network}_{model_type}"
        if cache_key in self._cache:
            logger.debug(f"Using cached model for {cache_key}")
            return self._cache[cache_key]
        
        network_dir = self.models_dir / network
        if not network_dir.exists():
            raise ValueError(f"No models found for network: {network}")
        
        pattern = f"{model_type}_{network}_v*.txt"
        models = sorted(network_dir.glob(pattern), reverse=True)
        
        if not models:
            raise ValueError(
                f"No models found for type={model_type}, network={network}"
            )
        
        latest_model_path = models[0]
        logger.info(f"Loading latest model: {latest_model_path.name}")
        
        model = self.load_model(latest_model_path)
        self._cache[cache_key] = model
        
        return model
    
    def load_model(
        self,
        model_path: Path
    ) -> Union[xgb.XGBClassifier, lgb.Booster]:
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        metadata = self.get_metadata(model_path)
        model_trainer = metadata.get('model_trainer', 'XGBoostTrainer')
        
        if 'XGBoost' in model_trainer:
            model = xgb.XGBClassifier()
            model.load_model(str(model_path))
            logger.info(f"Loaded XGBoost model from {model_path}")
        else:
            model = lgb.Booster(model_file=str(model_path))
            logger.info(f"Loaded LightGBM model from {model_path}")
        
        return model
    
    def get_metadata(self, model_path: Path) -> Dict[str, Any]:
        
        metadata_path = model_path.with_suffix('.json')
        
        if not metadata_path.exists():
            logger.warning(f"No metadata found for {model_path.name}")
            return {}
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_model_version(self, model_path: Path) -> str:
        
        metadata = self.get_metadata(model_path)
        return metadata.get('version', 'unknown')
    
    def clear_cache(self):
        
        self._cache.clear()
        logger.info("Model cache cleared")