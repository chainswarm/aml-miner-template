#!/usr/bin/env python3
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import ClientFactory, get_connection_params
from packages.scoring import RiskScoring


def main():
    parser = argparse.ArgumentParser(description="Risk scoring batch processing")
    parser.add_argument('--network', type=str, required=True, help='Network identifier (ethereum, bitcoin, torus, etc.)')
    parser.add_argument('--processing-date', type=str, required=True, help='Processing date (YYYY-MM-DD)')
    parser.add_argument('--window-days', type=int, default=7, help='Window days to filter (7, 30, 90, 195)')
    parser.add_argument('--models-dir', type=Path, default=None, help='Models directory (default: data/trained_models)')
    parser.add_argument(
        '--model-types',
        type=str,
        nargs='+',
        default=['alert_scorer', 'alert_ranker', 'cluster_scorer'],
        choices=['alert_scorer', 'alert_ranker', 'cluster_scorer'],
        help='Model types to run'
    )
    args = parser.parse_args()
    
    service_name = f'{args.network}-risk-scoring'
    setup_logger(service_name)
    load_dotenv()
    
    logger.info(
        "Initializing risk scoring",
        extra={
            "network": args.network,
            "processing_date": args.processing_date,
            "window_days": args.window_days,
            "model_types": args.model_types
        }
    )
    
    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)
    
    with client_factory.client_context() as client:
        scoring = RiskScoring(
            network=args.network,
            processing_date=args.processing_date,
            client=client,
            window_days=args.window_days,
            models_dir=args.models_dir,
            model_types=args.model_types
        )
        
        scoring.run()


if __name__ == "__main__":
    main()