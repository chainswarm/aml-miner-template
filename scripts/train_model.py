#!/usr/bin/env python3
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import ClientFactory, get_connection_params
from packages.training.model_training import ModelTraining


def main():
    parser = argparse.ArgumentParser(description="Model training")
    parser.add_argument('--network', type=str, required=True, help='Network identifier (ethereum, bitcoin, torus, etc.)')
    parser.add_argument('--start-date', type=str, required=True, help='Start processing_date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, required=True, help='End processing_date (YYYY-MM-DD)')
    parser.add_argument('--model-type', type=str, default='alert_scorer', choices=['alert_scorer', 'alert_ranker', 'cluster_scorer'], help='Type of model to train')
    parser.add_argument('--window-days', type=int, default=7, help='Window days to filter (7, 30, 90, 195)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory for models')
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


if __name__ == "__main__":
    main()