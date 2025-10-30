import argparse
from abc import ABC
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import ClientFactory, get_connection_params
from packages.training.feature_extraction import FeatureExtractor
from packages.training.feature_builder import FeatureBuilder
from packages.training.model_storage import ModelStorage
from packages.training.strategies import (
    LabelStrategy,
    ModelTrainer,
    AddressLabelStrategy,
    XGBoostTrainer
)

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class ModelTraining(ABC):
    
    def __init__(
        self,
        network: str,
        start_date: str,
        end_date: str,
        client,
        model_type: str = 'alert_scorer',
        window_days: int = 7,
        output_dir: Path = None,
        label_strategy: LabelStrategy = None,
        model_trainer: ModelTrainer = None
    ):
        self.network = network
        self.start_date = start_date
        self.end_date = end_date
        self.client = client
        self.model_type = model_type
        self.window_days = window_days
        
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'data' / 'trained_models' / network
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.label_strategy = label_strategy or AddressLabelStrategy()
        self.model_trainer = model_trainer or XGBoostTrainer()
        
        logger.info(
            f"Using label strategy: {self.label_strategy.__class__.__name__}, "
            f"model trainer: {self.model_trainer.__class__.__name__}"
        )
    
    def run(self):
        
        if terminate_event.is_set():
            logger.info("Termination requested before start")
            return
        
        logger.info(
            "Starting training workflow",
            extra={
                "network": self.network,
                "start_date": self.start_date,
                "end_date": self.end_date,
                "model_type": self.model_type,
                "window_days": self.window_days
            }
        )
        
        logger.info("Extracting training data from ClickHouse")
        extractor = FeatureExtractor(self.client)
        data = extractor.extract_training_data(
            start_date=self.start_date,
            end_date=self.end_date,
            window_days=self.window_days
        )
        
        if terminate_event.is_set():
            logger.warning("Termination requested after extraction")
            return
        
        logger.info("Deriving labels using strategy")
        alerts_with_labels = self.label_strategy.derive_labels(
            data['alerts'],
            data
        )
        
        if not self.label_strategy.validate_labels(alerts_with_labels):
            raise ValueError("Label validation failed")
        
        data['alerts'] = alerts_with_labels
        
        logger.info("Building feature matrix")
        builder = FeatureBuilder()
        X, y = builder.build_training_features(data)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after feature building")
            return
        
        logger.info("Getting sample weights")
        labeled_alerts = alerts_with_labels[alerts_with_labels['label'].notna()].copy()
        sample_weights = self.label_strategy.get_label_weights(labeled_alerts)
        
        logger.info("Training model")
        model = self.model_trainer.train(X, y, sample_weights)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after training")
            return
        
        logger.info("Evaluating model")
        metrics = self.model_trainer.evaluate(X, y)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after training")
            return
        
        logger.info("Saving model and metadata")
        storage = ModelStorage(self.output_dir, self.client)
        
        training_config = {
            'start_date': self.start_date,
            'end_date': self.end_date,
            'window_days': self.window_days,
            'num_samples': len(X),
            'positive_rate': float(y.mean()),
            'version': '1.0.0',
            'label_strategy': self.label_strategy.__class__.__name__,
            'model_trainer': self.model_trainer.__class__.__name__
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
                "auc": metrics.get('auc', 0.0),
                "pr_auc": metrics.get('pr_auc', 0.0)
            }
        )

