import argparse
import os
from abc import ABC
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import get_connection_params, ClientFactory, MigrateSchema, create_database


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
                logger.warning(f"No files found in S3 at {self.s3_prefix}")
                return 0

            expected_files = [
                f'alerts_{self.processing_date}_{self.days}d.parquet',
                f'features_{self.processing_date}_{self.days}d.parquet',
                f'clusters_{self.processing_date}_{self.days}d.parquet'
            ]
            
            expected_files_alt = [
                f'alerts_{self.processing_date}.parquet',
                f'features_{self.processing_date}.parquet',
                f'clusters_{self.processing_date}.parquet'
            ]

            all_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
            
            files_to_download = []
            for s3_key in all_files:
                filename = os.path.basename(s3_key)
                if filename in expected_files or filename in expected_files_alt:
                    files_to_download.append(s3_key)
            
            if not files_to_download:
                logger.error(
                    f"No matching files found for processing_date={self.processing_date}, days={self.days}",
                    extra={"expected": expected_files, "found": [os.path.basename(f) for f in all_files]}
                )
                raise ValueError(f"No parquet files found for {self.processing_date}")

            logger.info(f"Found {len(files_to_download)} files to download: {[os.path.basename(f) for f in files_to_download]}")

            downloaded_count = 0
            for s3_key in files_to_download:
                if terminate_event.is_set():
                    logger.warning("Termination requested during download")
                    return downloaded_count
                    
                try:
                    self._download_file(s3_key)
                    downloaded_count += 1
                except Exception as e:
                    logger.error(f"Failed to download {s3_key}: {e}")
                    raise

            logger.success(f"Download completed: {downloaded_count}/{len(files_to_download)} files downloaded to {self.local_dir}")
            return downloaded_count

        except ClientError as e:
            logger.error(f"S3 error while listing objects: {e}")
            raise

    def _validate_parquet_file(self, file_path: str, expected_table: str) -> bool:
        import pyarrow.parquet as pq
        
        try:
            parquet_file = pq.ParquetFile(file_path)
            schema = parquet_file.schema_arrow
            num_rows = parquet_file.metadata.num_rows
            
            logger.info(
                f"Parquet validation for {os.path.basename(file_path)}",
                extra={
                    "num_rows": num_rows,
                    "num_columns": len(schema),
                    "columns": [field.name for field in schema]
                }
            )
            
            if num_rows == 0:
                logger.error(f"Parquet file {file_path} is empty (0 rows)")
                return False
            
            required_columns = {
                'raw_alerts': ['alert_id', 'processing_date', 'window_days', 'address'],
                'raw_features': ['processing_date', 'address', 'feature_name', 'feature_value'],
                'raw_clusters': ['cluster_id', 'processing_date', 'window_days']
            }
            
            if expected_table in required_columns:
                file_columns = {field.name for field in schema}
                missing_columns = set(required_columns[expected_table]) - file_columns
                
                if missing_columns:
                    logger.error(
                        f"Missing required columns in {file_path}",
                        extra={"missing": list(missing_columns)}
                    )
                    return False
            
            logger.success(f"Parquet file validation passed: {os.path.basename(file_path)}")
            return True
            
        except Exception as e:
            logger.error(f"Parquet validation failed for {file_path}: {e}")
            return False

    def run(self):
        if terminate_event.is_set():
            logger.info("Termination requested before start")
            return

        logger.info(
            "Starting ingestion workflow",
            extra={
                "network": self.network,
                "processing_date": self.processing_date,
                "window_days": self.days
            }
        )
        
        logger.info("Step 1/6: Checking if data already exists")
        
        validation_query = f"""
            SELECT COUNT(DISTINCT table) as tables_with_data
            FROM (
                SELECT 'risk_scoring_raw_alerts' as table
                FROM risk_scoring_raw_alerts
                WHERE processing_date = '{self.processing_date}'
                  AND window_days = {self.days}
                LIMIT 1
                
                UNION ALL
                
                SELECT 'risk_scoring_raw_features' as table
                FROM risk_scoring_raw_features
                WHERE processing_date = '{self.processing_date}'
                LIMIT 1
                
                UNION ALL
                
                SELECT 'risk_scoring_raw_clusters' as table
                FROM risk_scoring_raw_clusters
                WHERE processing_date = '{self.processing_date}'
                  AND window_days = {self.days}
                LIMIT 1
            )
        """
        
        result = self.client.query(validation_query)
        tables_with_data = result.result_rows[0][0] if result.result_rows else 0
        
        if tables_with_data == 3:
            logger.success(f"Data already fully ingested for {self.processing_date} (window: {self.days} days)")
            return
        
        logger.info(f"Found data in {tables_with_data}/3 tables")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after data validation check")
            return
        
        if tables_with_data > 0:
            logger.info("Step 2/6: Cleaning up partial data")
            logger.warning(f"Partial data detected ({tables_with_data}/3 tables). Cleaning up...")
            
            cleanup_queries = [
                f"ALTER TABLE risk_scoring_raw_alerts DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
                f"ALTER TABLE risk_scoring_raw_features DELETE WHERE processing_date = '{self.processing_date}'",
                f"ALTER TABLE risk_scoring_raw_clusters DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}"
            ]
            
            for query in cleanup_queries:
                if terminate_event.is_set():
                    logger.warning("Termination requested during cleanup")
                    return
                self.client.command(query)
            
            logger.success("Cleanup complete")
        else:
            logger.info("Step 2/6: No cleanup needed (no existing data)")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after cleanup")
            return
        
        logger.info("Step 3/6: Downloading files from S3")
        logger.info(f"S3 source: s3://{self.bucket}/{self.s3_prefix}")
        logger.info(f"Local destination: {self.local_dir}")
        
        downloaded_count = self._download_all()
        
        if downloaded_count == 0:
            raise ValueError("No files downloaded from S3")
        
        logger.success(f"Downloaded {downloaded_count} files")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after download")
            return
        
        logger.info("Step 4/6: Validating parquet files")
        
        ingestion_files = {}
        for table, base_name in [
            ('risk_scoring_raw_alerts', 'alerts'),
            ('risk_scoring_raw_features', 'features'),
            ('risk_scoring_raw_clusters', 'clusters')
        ]:
            file_with_days = f'{base_name}_{self.processing_date}_{self.days}d.parquet'
            file_without_days = f'{base_name}_{self.processing_date}.parquet'
            
            path_with_days = os.path.join(self.local_dir, file_with_days)
            path_without_days = os.path.join(self.local_dir, file_without_days)
            
            if os.path.exists(path_with_days):
                ingestion_files[table] = file_with_days
            elif os.path.exists(path_without_days):
                ingestion_files[table] = file_without_days
            else:
                raise FileNotFoundError(
                    f"Expected parquet file not found. Tried: {file_with_days}, {file_without_days}"
                )
        
        validation_failed = []
        for table, filename in ingestion_files.items():
            if terminate_event.is_set():
                logger.warning("Termination requested during validation")
                return
                
            file_path = os.path.join(self.local_dir, filename)
            
            if not self._validate_parquet_file(file_path, table):
                validation_failed.append(filename)
        
        if validation_failed:
            raise ValueError(f"Parquet validation failed for: {', '.join(validation_failed)}")
        
        logger.success("All parquet files validated successfully")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after validation")
            return
        
        logger.info("Step 5/6: Ingesting data into ClickHouse")
        logger.info(f"Target: {self.network} database")
        
        for table, filename in ingestion_files.items():
            if terminate_event.is_set():
                logger.warning(f"Termination requested during ingestion (completed: {list(ingestion_files.keys())[:list(ingestion_files.keys()).index(table)]})")
                return
                
            file_path = os.path.join(self.local_dir, filename)
            
            logger.info(f"Ingesting {filename} into {table}")
            
            try:
                self.client.insert_file(
                    table=table,
                    file_path=file_path,
                    fmt='Parquet'
                )
                
                logger.success(f"Ingested {filename} into {table}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {filename} into {table}: {e}")
                raise
        
        logger.success("All data ingested successfully")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after ingestion")
            return
        
        logger.info("Step 6/6: Verifying ingestion")
        
        verify_query = f"""
            SELECT
                'risk_scoring_raw_alerts' as table, COUNT(*) as count
            FROM raw_alerts
            WHERE processing_date = '{self.processing_date}'
              AND window_days = {self.days}
            
            UNION ALL
            
            SELECT
                'risk_scoring_raw_features' as table, COUNT(*) as count
            FROM raw_features
            WHERE processing_date = '{self.processing_date}'
            
            UNION ALL
            
            SELECT
                'risk_scoring_raw_clusters' as table, COUNT(*) as count
            FROM raw_clusters
            WHERE processing_date = '{self.processing_date}'
              AND window_days = {self.days}
        """
        
        verify_result = self.client.query(verify_query)
        
        total_records = 0
        for row in verify_result.result_rows:
            table_name, count = row
            total_records += count
            logger.info(f"{table_name}: {count:,} records")
        
        if total_records == 0:
            raise ValueError("Ingestion verification failed: No records found in database")
        
        logger.success(
            "Ingestion workflow completed successfully",
            extra={
                "total_records": total_records,
                "network": self.network,
                "processing_date": self.processing_date,
                "window_days": self.days
            }
        )

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data sync")
    parser.add_argument('--network', type=str, required=True)
    parser.add_argument('--processing-date', type=str, required=True)
    parser.add_argument('--days', type=int, required=True)
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

    s3_endpoint = os.getenv('RISK_SCORING_S3_ENDPOINT')
    s3_bucket = os.getenv('RISK_SCORING_S3_BUCKET')
    s3_region = os.getenv('RISK_SCORING_S3_REGION', 'nl-ams')

    if not all([s3_endpoint, s3_bucket]):
        logger.critical("Missing required S3 configuration (RISK_SCORING_S3_ENDPOINT, RISK_SCORING_S3_BUCKET)")
        import sys
        sys.exit(1)

    create_database(connection_params)

    with client_factory.client_context() as client:
        migrate_schema = MigrateSchema(client)
        migrate_schema.run_migrations()

        from botocore import UNSIGNED
        from botocore.config import Config
        
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