import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union
import lightgbm as lgb
import xgboost as xgb
from loguru import logger
from clickhouse_connect.driver import Client


class ModelStorage:
    
    def __init__(self, output_dir: Path, client: Optional[Client] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.client = client
    
    def cleanup_old_models(
        self,
        model_type: str,
        network: str,
        version: str,
        start_date: str,
        end_date: str,
        window_days: int,
        keep_latest: int = 1
    ):
        
        pattern = (
            f"{model_type}_{network}_v{version}_"
            f"{start_date}_{end_date}_w{window_days}d_*.txt"
        )
        
        models = sorted(self.output_dir.glob(pattern), reverse=True)
        
        if len(models) <= keep_latest:
            logger.info(f"No old models to clean up (found {len(models)}, keeping {keep_latest})")
            return
        
        models_to_delete = models[keep_latest:]
        
        for model_path in models_to_delete:
            model_path.unlink()
            logger.info(f"Deleted old model: {model_path.name}")
            
            metadata_path = model_path.with_suffix('.json')
            if metadata_path.exists():
                metadata_path.unlink()
                logger.info(f"Deleted metadata: {metadata_path.name}")
        
        if self.client:
            for model_path in models_to_delete:
                model_id = model_path.stem
                self.client.command(
                    f"ALTER TABLE trained_models DELETE WHERE model_id = '{model_id}'"
                )
                logger.info(f"Deleted model metadata from ClickHouse: {model_id}")
            
            self.client.command("OPTIMIZE TABLE trained_models FINAL")
        
        logger.success(f"Cleaned up {len(models_to_delete)} old model(s)")
    
    def save_model(
        self,
        model: Union[lgb.Booster, xgb.XGBClassifier],
        model_type: str,
        network: str,
        metrics: Dict[str, float],
        training_config: Dict[str, Any]
    ) -> Path:
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        version = training_config.get('version', '1.0.0')
        window_days = training_config.get('window_days', 7)
        start_date = training_config.get('start_date', 'unknown')
        end_date = training_config.get('end_date', 'unknown')
        
        model_filename = (
            f"{model_type}_{network}_v{version}_"
            f"{start_date}_{end_date}_w{window_days}d_{timestamp}.txt"
        )
        model_path = self.output_dir / model_filename
        
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")
        
        metadata = self._create_metadata(
            model=model,
            model_type=model_type,
            network=network,
            version=version,
            timestamp=timestamp,
            metrics=metrics,
            training_config=training_config
        )
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.success(f"Metadata saved to {metadata_path}")
        
        if self.client:
            self._store_in_clickhouse(model_path, metadata)
        
        return model_path
    
    def _create_metadata(
        self,
        model: Union[lgb.Booster, xgb.XGBClassifier],
        model_type: str,
        network: str,
        version: str,
        timestamp: str,
        metrics: Dict[str, float],
        training_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        
        if isinstance(model, xgb.XGBClassifier):
            num_features = model.n_features_in_
            hyperparameters = model.get_params()
            feature_names = model.get_booster().feature_names if hasattr(model.get_booster(), 'feature_names') else []
            num_trees = model.n_estimators
            best_iteration = getattr(model, 'best_iteration', num_trees)
        else:
            num_features = model.num_feature()
            hyperparameters = model.params
            feature_names = model.feature_name()
            num_trees = model.num_trees()
            best_iteration = model.best_iteration
        
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
                'num_features': num_features,
                'positive_rate': training_config.get('positive_rate', 0.0)
            },
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'feature_names': feature_names,
            'num_trees': num_trees,
            'best_iteration': best_iteration
        }
    
    def _store_in_clickhouse(self, model_path: Path, metadata: Dict[str, Any]):
        
        logger.info("Storing model metadata in ClickHouse")
        
        try:
            start_date = metadata['training_period']['start_date']
            end_date = metadata['training_period']['end_date']
            
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
            self.client.insert(
                'trained_models',
                [[
                    metadata['model_id'],
                    metadata['model_type'],
                    metadata['version'],
                    metadata['network'],
                    start_date,
                    end_date,
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
                ]],
                column_names=[
                    'model_id', 'model_type', 'version', 'network',
                    'training_start_date', 'training_end_date', 'created_at',
                    'model_path', 'metrics_json', 'hyperparameters_json',
                    'feature_names', 'num_samples', 'num_features',
                    'test_auc', 'cv_auc_mean', 'cv_auc_std'
                ]
            )
            
            logger.success("Model metadata stored in ClickHouse")
        
        except Exception as e:
            logger.error(f"Failed to store metadata in ClickHouse: {e}")
            raise
    
    def load_model(self, model_path: Path) -> lgb.Booster:
        
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