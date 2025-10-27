import argparse
import os
from abc import ABC
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger
from alert_scoring import setup_logger, terminate_event
from alert_scoring.storage import get_connection_params, ClientFactory, MigrateSchema
from alert_scoring.storage.repositories import AlertsRepository, ClusterScoresRepository, ClustersRepository, FeaturesRepository, MetadataRepository, RankingsRepository


class SOTDataIngestion(ABC):

    def __init__(self, network, processing_date, days, client, s3_client, bucket):
        self.network = network
        self.processing_date = processing_date
        self.days = days
        self.client = client

        self.s3_client = s3_client
        self.bucket = bucket
        self.network = network
        self.local_dir = os.path.join('data', 'input', network)
        self.s3_prefix = f"alerts/{network}/"
        os.makedirs(self.local_dir, exist_ok=True)

    def _download_all(self) -> int:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.s3_prefix
            )

            if 'Contents' not in response:
                logger.warning("No files found in S3")
                return 0

            files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]

            logger.info(f"Found {len(files)} files to download")

            downloaded_count = 0
            for s3_key in files:
                try:
                    self._download_file(s3_key)
                    downloaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to download {s3_key}: {e}")
                    raise

            logger.success(f"Download completed: {downloaded_count} files downloaded to {self.local_dir}")
            return downloaded_count

        except ClientError as e:
            logger.error(f"Failed to list S3 objects: {e}")
            raise

    def run(self):

        if terminate_event.is_set():
            return

        logger.info("Checking ingestion status")
        # client.query() query clickhouse if we have any data for given processing date, days for tables: raw_features, raw_clusters, raw_alerts; use UNION and count number-  we expect 3, if less then remove all data by processing date and days, we are are sure we operate on fresh data

        # if we have ingested data already, we return
        # if we did not ingested data into clickhouse, then we download files from s3
        logger.info(f"Downloading data from {self.s3_prefix} to {self.local_dir}")
        self._download_all()


        logger.info(f"Starting ingestion on {self.processing_date} for {self.days} days of {self.network} network")
        #we have downloaded data from s3 without any errors, and we know our clickhouse is ready to take the ingestion
        # ingest data from parquet fiels forom disk by using self.client (clickhouse connect)

        logger.success("Ingestion complete")

    def _download_file(self, s3_key: str) -> str:
        filename = os.path.basename(s3_key)
        local_path = os.path.join(self.local_dir, filename)

        logger.info(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")

        try:
            self.s3_client.download_file(self.bucket, s3_key, local_path)

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.success(f"Downloaded {filename} ({file_size_mb:.2f} MB)")
            return local_path

        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

if "__main__" == __name__:
    parser = argparse.ArgumentParser(description="Data sync")
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--processing-date', type=int, default=7)
    parser.add_argument('--days', type=int, default=7)
    args = parser.parse_args()

    service_name = f'{args.network}-{args.processing_date}-{args.days}-data-sync'
    setup_logger(service_name)
    load_dotenv()

    logger.info(
        "Initializing data sync",
        extra={
            "network": args.network,
            "processing_date": args.processing_date,
            "days": args.days,
        }
    )

    connection_params = get_connection_params(args.network)
    client_factory = ClientFactory(connection_params)

    s3_endpoint = os.getenv('S3_ENDPOINT')
    s3_access_key = os.getenv('S3_ACCESS_KEY')
    s3_secret_key = os.getenv('S3_SECRET_KEY')
    s3_bucket = os.getenv('S3_BUCKET')
    s3_region = os.getenv('S3_REGION', 'us-east-1')

    if not all([s3_endpoint, s3_access_key, s3_secret_key, s3_bucket]):
        logger.critical("Missing required S3 configuration. Provide via command line or .env file")
        import sys
        sys.exit(1)

    with client_factory.client_context() as client:
        migrate_schema = MigrateSchema(client)
        migrate_schema.create_database(args.network)
        migrate_schema.run_migrations()

        s3_client = boto3.client(
            's3',
            endpoint_url=s3_endpoint,
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name=s3_region
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