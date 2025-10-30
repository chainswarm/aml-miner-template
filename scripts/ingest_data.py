#!/usr/bin/env python3
import argparse
import os
import sys
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger
from packages.storage import ClientFactory, get_connection_params, MigrateSchema, create_database
from packages.ingestion.sot_ingestion import SOTDataIngestion


def main():
    parser = argparse.ArgumentParser(description="Data ingestion from S3 to ClickHouse")
    parser.add_argument('--network', type=str, required=True, help='Network identifier (ethereum, bitcoin, torus, etc.)')
    parser.add_argument('--processing-date', type=str, required=True, help='Processing date (YYYY-MM-DD)')
    parser.add_argument('--days', type=int, required=True, help='Window days (7, 30, 90, 195)')
    args = parser.parse_args()

    service_name = f'{args.network}-{args.processing_date}-{args.days}-data-sync'
    setup_logger(service_name)
    load_dotenv()

    logger.info(
        "Initializing data ingestion",
        extra={
            "network": args.network,
            "processing_date": args.processing_date,
            "days": args.days,
        }
    )

    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)

    s3_endpoint = os.getenv('RISK_SCORING_S3_ENDPOINT')
    s3_bucket = os.getenv('RISK_SCORING_S3_BUCKET')
    s3_region = os.getenv('RISK_SCORING_S3_REGION', 'nl-ams')

    if not all([s3_endpoint, s3_bucket]):
        logger.critical("Missing required S3 configuration (RISK_SCORING_S3_ENDPOINT, RISK_SCORING_S3_BUCKET)")
        sys.exit(1)

    create_database(connection_params)

    with client_factory.client_context() as client:
        migrate_schema = MigrateSchema(client)
        migrate_schema.run_migrations()

        s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            region_name=s3_region,
            config=Config(signature_version=UNSIGNED)
        )

        logger.info(f"Connected to S3: {s3_endpoint}")
        
        data_sync = SOTDataIngestion(
            network=args.network,
            processing_date=args.processing_date,
            days=args.days,
            client=client,
            s3_client=s3_client,
            bucket=s3_bucket,
        )

        data_sync.run()


if __name__ == "__main__":
    main()