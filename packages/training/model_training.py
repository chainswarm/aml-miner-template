import argparse
from abc import ABC
from pathlib import Path
from dotenv import load_dotenv
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
        window_days: int = 7,
        output_dir: Path = None
    ):
        self.network = network
        self.start_date = start_date
        self.end_date = end_date
        self.client = client
        self.model_type = model_type
        self.window_days = window_days
        
        if output_dir is None:
            output_dir = PROJECT_ROOT / 'trained_models' / network
        
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
        
        logger.info("Building feature matrix")
        builder = FeatureBuilder()
        X, y = builder.build_training_features(data)
        
        if terminate_event.is_set():
            logger.warning("Termination requested after feature building")
            return
        
        logger.info("Training model")
        trainer = ModelTrainer(model_type=self.model_type)
        model, metrics = trainer.train(X, y, cv_folds=5)
        
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
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--network', type=str, required=True,
                       help='Network identifier (ethereum, bitcoin, etc.)')
    parser.add_argument('--start-date', type=str, required=True,
                       help='Start processing_date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True,
                       help='End processing_date (YYYY-MM-DD)')
    parser.add_argument('--model-type', type=str, default='alert_scorer',
                       choices=['alert_scorer', 'alert_ranker', 'cluster_scorer'],
                       help='Type of model to train')
    parser.add_argument('--window-days', type=int, default=7,
                       help='Window days to filter (7, 30, 90)')
    parser.add_argument('--output-dir', type=Path, default=None,
                       help='Output directory for models')
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
            "model_type": args.model_type,
            "window_days": args.window_days
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
            window_days=args.window_days,
            output_dir=args.output_dir
        )
        
        training.run()