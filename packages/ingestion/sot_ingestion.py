import argparse
import os
import json
import hashlib
from abc import ABC
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from loguru import logger
from packages import setup_logger, terminate_event
from packages.storage import get_connection_params, ClientFactory, MigrateSchema, create_database

PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


class SOTDataIngestion(ABC):

    def __init__(self, network, processing_date, days, client, s3_client, bucket):
        self.network = network
        self.processing_date = processing_date
        self.days = days
        self.client = client

        self.s3_client = s3_client
        self.bucket = bucket
        self.network = network
        self.local_dir = PROJECT_ROOT / 'data' / 'input' / 'risk-scoring' / 'snapshots' / network / processing_date / str(days)
        self.s3_prefix = f"snapshots/{network}/{processing_date}/{days}"
        os.makedirs(self.local_dir, exist_ok=True)

    def _calculate_md5(self, file_path: str) -> str:
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
        return md5_hash.hexdigest()

    def _verify_checksum(self, file_path: str, expected_checksum: str) -> bool:
        actual_checksum = self._calculate_md5(file_path)
        if actual_checksum != expected_checksum:
            logger.error(
                f"Checksum mismatch for {os.path.basename(file_path)}",
                extra={
                    "expected": expected_checksum,
                    "actual": actual_checksum
                }
            )
            return False
        logger.success(f"Checksum verified for {os.path.basename(file_path)}")
        return True

    def _download_all(self) -> int:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.s3_prefix
            )

            if 'Contents' not in response:
                logger.warning(f"No files found in S3 at {self.s3_prefix}")
                return 0

            meta_key = f"{self.s3_prefix}/META.json"
            meta_file_path = self.local_dir / "META.json"
            
            logger.info("Downloading META.json")
            self._download_file(meta_key)
            
            if not os.path.exists(meta_file_path):
                raise FileNotFoundError(f"META.json not found after download: {meta_file_path}")
            
            with open(meta_file_path, 'r') as f:
                metadata = json.load(f)
            
            logger.info(f"Loaded metadata with {len(metadata.get('files', {}))} file entries")
            
            all_files = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.parquet')]
            
            files_to_download = []
            for s3_key in all_files:
                files_to_download.append(s3_key)
            
            if not files_to_download:
                raise ValueError(f"No parquet files found for {self.processing_date}, days={self.days}")

            logger.info(f"Found {len(files_to_download)} files to download: {[os.path.basename(f) for f in files_to_download]}")

            downloaded_count = 0
            for s3_key in files_to_download:
                if terminate_event.is_set():
                    logger.warning("Termination requested during download")
                    return downloaded_count
                    
                try:
                    local_path = self._download_file(s3_key)
                    
                    filename = os.path.basename(s3_key)
                    if filename in metadata.get('files', {}):
                        expected_checksum = metadata['files'][filename]['md5']
                        if not self._verify_checksum(local_path, expected_checksum):
                            raise ValueError(f"Checksum verification failed for {filename}")
                    else:
                        logger.warning(f"No checksum found in META.json for {filename}")
                    
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
                'raw_features': ['processing_date', 'window_days', 'address'],
                'raw_clusters': ['cluster_id', 'processing_date', 'window_days'],
                'raw_money_flows': ['from_address', 'to_address', 'processing_date', 'window_days']
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
        
        logger.info("Checking if data already exists")
        
        validation_query = f"""
            SELECT COUNT(DISTINCT table) as tables_with_data
            FROM (
                SELECT 'raw_alerts' as table
                FROM raw_alerts
                WHERE processing_date = '{self.processing_date}'
                  AND window_days = {self.days}
                LIMIT 1
                
                UNION ALL
                
                SELECT 'raw_features' as table
                FROM raw_features
                WHERE processing_date = '{self.processing_date}'
                  AND window_days = {self.days}
                LIMIT 1
                
                UNION ALL
                
                SELECT 'raw_clusters' as table
                FROM raw_clusters
                WHERE processing_date = '{self.processing_date}'
                  AND window_days = {self.days}
                LIMIT 1
                
                UNION ALL
                
                SELECT 'raw_money_flows' as table
                FROM raw_money_flows
                WHERE processing_date = '{self.processing_date}'
                  AND window_days = {self.days}
                LIMIT 1
            )
        """
        
        result = self.client.query(validation_query)
        tables_with_data = result.result_rows[0][0] if result.result_rows else 0
        
        if tables_with_data == 4:
            logger.success(f"Data already fully ingested for {self.processing_date} (window: {self.days} days)")
            return
        
        logger.info(f"Found data in {tables_with_data}/4 tables")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after data validation check")
            return
        
        if tables_with_data > 0:
            logger.info("Cleaning up partial data")
            logger.warning(f"Partial data detected ({tables_with_data}/4 tables). Cleaning up...")
            
            cleanup_queries = [
                f"ALTER TABLE raw_alerts DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
                f"ALTER TABLE raw_features DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
                f"ALTER TABLE raw_clusters DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}",
                f"ALTER TABLE raw_money_flows DELETE WHERE processing_date = '{self.processing_date}' AND window_days = {self.days}"
            ]
            
            for query in cleanup_queries:
                if terminate_event.is_set():
                    logger.warning("Termination requested during cleanup")
                    return
                self.client.command(query)
            
            logger.success("Cleanup complete")
        else:
            logger.info("No cleanup needed (no existing data)")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after cleanup")
            return
        
        logger.info("Downloading files from S3")
        logger.info(f"S3 source: s3://{self.bucket}/{self.s3_prefix}")
        logger.info(f"Local destination: {self.local_dir}")
        
        downloaded_count = self._download_all()
        
        if downloaded_count == 0:
            raise ValueError("No files downloaded from S3")
        
        logger.success(f"Downloaded {downloaded_count} files")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after download")
            return
        
        logger.info("Downloading address labels")
        has_address_labels = self._download_address_labels()
        
        if terminate_event.is_set():
            logger.warning("Termination requested after address labels download")
            return
        
        logger.info("Validating parquet files")
        
        ingestion_files = {}
        for table, base_name in [
            ('raw_alerts', 'alerts'),
            ('raw_features', 'features'),
            ('raw_clusters', 'clusters'),
            ('raw_money_flows', 'money_flows')
        ]:
            filename = f'{base_name}.parquet'
            file_path = self.local_dir / filename
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Expected parquet file not found: {file_path}"
                )
            
            ingestion_files[table] = filename
        
        validation_failed = []
        for table, filename in ingestion_files.items():
            if terminate_event.is_set():
                logger.warning("Termination requested during validation")
                return
                
            file_path = self.local_dir / filename
            
            if not self._validate_parquet_file(str(file_path), table):
                validation_failed.append(filename)
        
        if validation_failed:
            raise ValueError(f"Parquet validation failed for: {', '.join(validation_failed)}")
        
        logger.success("All parquet files validated successfully")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after validation")
            return
        
        logger.info("Ingesting data into ClickHouse")
        logger.info(f"Target: {self.network} database")
        
        for table, filename in ingestion_files.items():
            if terminate_event.is_set():
                logger.warning(f"Termination requested during ingestion (completed: {list(ingestion_files.keys())[:list(ingestion_files.keys()).index(table)]})")
                return
                
            file_path = self.local_dir / filename
            
            logger.info(f"Ingesting {filename} into {table}")
            
            try:
                import pandas as pd
                df = pd.read_parquet(file_path)
                
                if 'processing_date' in df.columns:
                    df['processing_date'] = pd.to_datetime(df['processing_date'])
                
                self.client.insert_df(table=table, df=df)
                
                logger.success(f"Ingested {filename} into {table}")
                
            except Exception as e:
                logger.error(f"Failed to ingest {filename} into {table}: {e}")
                raise
        
        logger.success("All data ingested successfully")
        
        if terminate_event.is_set():
            logger.warning("Termination requested after ingestion")
            return
        
        if has_address_labels:
            logger.info("Ingesting address labels")
            self._ingest_address_labels()
        
        if terminate_event.is_set():
            logger.warning("Termination requested after address labels ingestion")
            return
        
        logger.info("Verifying ingestion")
        
        verify_query = f"""
            SELECT
                'raw_alerts' as table, COUNT(*) as count
            FROM raw_alerts
            WHERE processing_date = '{self.processing_date}'
              AND window_days = {self.days}
            
            UNION ALL
            
            SELECT
                'raw_features' as table, COUNT(*) as count
            FROM raw_features
            WHERE processing_date = '{self.processing_date}'
              AND window_days = {self.days}
            
            UNION ALL
            
            SELECT
                'raw_clusters' as table, COUNT(*) as count
            FROM raw_clusters
            WHERE processing_date = '{self.processing_date}'
              AND window_days = {self.days}
            
            UNION ALL
            
            SELECT
                'raw_money_flows' as table, COUNT(*) as count
            FROM raw_money_flows
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
        local_path = PROJECT_ROOT / 'data' / 'input' / 'risk-scoring' / s3_key
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        logger.info(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")

        try:
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))

            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.success(f"Downloaded {os.path.basename(s3_key)} ({file_size_mb:.2f} MB)")
            return str(local_path)

        except ClientError as e:
            logger.error(f"Failed to download {s3_key}: {e}")
            raise

    def _download_address_labels(self) -> bool:
        s3_key = f"address-labels/{self.network}_address_labels.parquet"
        local_path = PROJECT_ROOT / 'data' / 'input' / 'risk-scoring' / 'address-labels' / f'{self.network}_address_labels.parquet'
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            logger.info(f"Downloading address labels from s3://{self.bucket}/{s3_key}")
            self.s3_client.download_file(self.bucket, s3_key, str(local_path))
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.success(f"Downloaded {self.network}_address_labels.parquet ({file_size_mb:.2f} MB)")
            return True
            
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                logger.warning(f"Address labels file not found at {s3_key}")
                return False
            logger.error(f"Failed to download address labels: {e}")
            raise

    def _ingest_address_labels(self):
        file_path = PROJECT_ROOT / 'data' / 'input' / 'risk-scoring' / 'address-labels' / f'{self.network}_address_labels.parquet'
        
        if not os.path.exists(file_path):
            logger.warning("Address labels file not found, skipping ingestion")
            return
        
        logger.info(f"Ingesting {self.network}_address_labels.parquet into raw_address_labels")
        
        try:
            import pandas as pd
            df = pd.read_parquet(file_path)
            
            if df.empty:
                logger.warning("Address labels file is empty, skipping ingestion")
                return
            
            df['processing_date'] = pd.to_datetime(self.processing_date)
            df['window_days'] = self.days
            
            if 'network' not in df.columns:
                df['network'] = self.network
            
            columns_to_drop = ['created_timestamp', 'updated_timestamp']
            df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
            
            self.client.insert_df(table='raw_address_labels', df=df)
            
            logger.success(f"Ingested {len(df):,} address label records into raw_address_labels")
            
        except Exception as e:
            logger.error(f"Failed to ingest address labels: {e}")
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