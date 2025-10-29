import argparse
from datetime import datetime
import pandas as pd
from loguru import logger
import clickhouse_connect

from alert_scoring.storage import get_connection_params, ClientFactory
from alert_scoring.storage.repositories import (
    AlertsRepository,
    FeaturesRepository,
    ClustersRepository,
    MetadataRepository
)


def validate_connection(host: str, port: int, database: str) -> bool:
    try:
        client = clickhouse_connect.get_client(host=host, port=port, database=database)
        client.ping()
        logger.info(f"Connection successful to {host}:{port}/{database}")
        return True
    except Exception as e:
        logger.error(f"Failed to connect to {host}:{port}/{database}: {e}")
        return False


def download_alerts_from_sot(sot_client, processing_date: str, network: str, sot_database: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {sot_database}.analyzers_alerts
    WHERE network = '{network}'
      AND DATE(detected_at) = '{processing_date}'
    ORDER BY alert_id
    """
    
    logger.info(f"Downloading alerts from SOT for {processing_date}/{network}")
    result = sot_client.query(query)
    
    if not result.result_rows:
        logger.warning(f"No alerts found in SOT for {processing_date}/{network}")
        return pd.DataFrame()
    
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    logger.info(f"Downloaded {len(df)} alerts from SOT")
    
    return df


def download_features_from_sot(sot_client, processing_date: str, network: str, sot_database: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {sot_database}.analyzers_features
    WHERE network = '{network}'
      AND DATE(detected_at) = '{processing_date}'
    ORDER BY feature_id
    """
    
    logger.info(f"Downloading features from SOT for {processing_date}/{network}")
    result = sot_client.query(query)
    
    if not result.result_rows:
        logger.warning(f"No features found in SOT for {processing_date}/{network}")
        return pd.DataFrame()
    
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    logger.info(f"Downloaded {len(df)} features from SOT")
    
    return df


def download_clusters_from_sot(sot_client, processing_date: str, network: str, sot_database: str) -> pd.DataFrame:
    query = f"""
    SELECT *
    FROM {sot_database}.analyzers_alert_clusters
    WHERE network = '{network}'
      AND DATE(detected_at) = '{processing_date}'
    ORDER BY cluster_id
    """
    
    logger.info(f"Downloading clusters from SOT for {processing_date}/{network}")
    result = sot_client.query(query)
    
    if not result.result_rows:
        logger.warning(f"No clusters found in SOT for {processing_date}/{network}")
        return pd.DataFrame()
    
    df = pd.DataFrame(result.result_rows, columns=result.column_names)
    logger.info(f"Downloaded {len(df)} clusters from SOT")
    
    return df


def validate_dataframe(df: pd.DataFrame, name: str, required_fields: list) -> bool:
    if df.empty:
        logger.warning(f"{name} dataframe is empty")
        return True
    
    missing_fields = [field for field in required_fields if field not in df.columns]
    if missing_fields:
        raise ValueError(f"{name} missing required fields: {missing_fields}")
    
    logger.info(f"{name} validation passed: {len(df)} rows, {len(df.columns)} columns")
    return True


def check_existing_data(connection_params: dict, processing_date: str, network: str) -> bool:
    with ClientFactory(connection_params).client_context() as client:
        alerts_repo = AlertsRepository(client)
        existing_alerts = alerts_repo.get_alerts(processing_date, network)
        
        if existing_alerts:
            logger.warning(f"Data already exists for {processing_date}/{network}: {len(existing_alerts)} alerts found")
            return True
        
        return False


def download_from_sot(
    processing_date: str,
    network: str,
    sot_host: str,
    sot_port: int,
    sot_database: str,
    dry_run: bool = False,
    force: bool = False
):
    logger.info(f"Starting download from SOT for {processing_date}/{network}")
    logger.info(f"SOT: {sot_host}:{sot_port}/{sot_database}")
    
    connection_params = get_connection_params(network)
    
    if not validate_connection(sot_host, sot_port, sot_database):
        raise ConnectionError("Failed to connect to SOT ClickHouse")
    
    local_host = connection_params['host']
    local_port = connection_params['port']
    local_database = connection_params['database']
    
    if not validate_connection(local_host, local_port, local_database):
        raise ConnectionError("Failed to connect to local ClickHouse")
    
    if not force and check_existing_data(connection_params, processing_date, network):
        logger.error("Data already exists. Use --force to overwrite")
        raise ValueError("Data already exists for this date/network combination")
    
    sot_client = clickhouse_connect.get_client(
        host=sot_host,
        port=sot_port,
        database=sot_database
    )
    
    try:
        alerts_df = download_alerts_from_sot(sot_client, processing_date, network, sot_database)
        features_df = download_features_from_sot(sot_client, processing_date, network, sot_database)
        clusters_df = download_clusters_from_sot(sot_client, processing_date, network, sot_database)
        
        validate_dataframe(alerts_df, "Alerts", ['alert_id', 'address', 'typology_type'])
        validate_dataframe(features_df, "Features", ['feature_id', 'alert_id'])
        validate_dataframe(clusters_df, "Clusters", ['cluster_id'])
        
        if alerts_df.empty:
            raise ValueError(f"No alerts found for {processing_date}/{network} in SOT")
        
        if features_df.empty:
            raise ValueError(f"No features found for {processing_date}/{network} in SOT")
        
        logger.info(f"Download summary:")
        logger.info(f"  Alerts: {len(alerts_df)}")
        logger.info(f"  Features: {len(features_df)}")
        logger.info(f"  Clusters: {len(clusters_df)}")
        
        if dry_run:
            logger.info("DRY RUN: Data would be downloaded but not stored")
            logger.info(f"Alerts columns: {list(alerts_df.columns)}")
            logger.info(f"Features columns: {list(features_df.columns)}")
            if not clusters_df.empty:
                logger.info(f"Clusters columns: {list(clusters_df.columns)}")
            return
        
        logger.info("Storing data in local ClickHouse...")
        
        with ClientFactory(connection_params).client_context() as client:
            alerts_repo = AlertsRepository(client)
            features_repo = FeaturesRepository(client)
            clusters_repo = ClustersRepository(client)
            metadata_repo = MetadataRepository(client)
            
            alerts_repo.insert_alerts(alerts_df, processing_date, network)
            features_repo.insert_features(features_df, processing_date, network)
            
            if not clusters_df.empty:
                clusters_repo.insert_clusters(clusters_df, processing_date, network)
            
            batch_info = {
                'source': 'SOT ClickHouse',
                'input_counts': {
                    'alerts': len(alerts_df),
                    'features': len(features_df),
                    'clusters': len(clusters_df)
                },
                'output_counts': {
                    'alert_scores': 0,
                    'alert_rankings': 0,
                    'cluster_scores': 0
                },
                'latencies_ms': {
                    'alert_scoring': 0,
                    'alert_ranking': 0,
                    'cluster_scoring': 0,
                    'total': 0
                },
                'model_versions': {
                    'alert_scorer': '',
                    'alert_ranker': '',
                    'cluster_scorer': ''
                },
                'processed_at': datetime.utcnow(),
                'status': 'DOWNLOADED'
            }
            
            metadata_repo.insert_metadata(processing_date, network, batch_info)
            
            logger.info(f"Successfully downloaded and stored data for {processing_date}/{network}")
            logger.info(f"  {len(alerts_df)} alerts")
            logger.info(f"  {len(features_df)} features")
            logger.info(f"  {len(clusters_df)} clusters")
            
    except Exception as e:
        logger.error(f"Error during download: {e}")
        
        try:
            with ClientFactory(connection_params).client_context() as client:
                metadata_repo = MetadataRepository(client)
                
                error_info = {
                    'source': 'SOT ClickHouse',
                    'input_counts': {'alerts': 0, 'features': 0, 'clusters': 0},
                    'output_counts': {'alert_scores': 0, 'alert_rankings': 0, 'cluster_scores': 0},
                    'latencies_ms': {'alert_scoring': 0, 'alert_ranking': 0, 'cluster_scoring': 0, 'total': 0},
                    'model_versions': {'alert_scorer': '', 'alert_ranker': '', 'cluster_scorer': ''},
                    'processed_at': datetime.utcnow(),
                    'status': 'DOWNLOAD_FAILED',
                    'error_message': str(e)
                }
                
                metadata_repo.insert_metadata(processing_date, network, error_info)
                
        except Exception as meta_error:
            logger.error(f"Failed to write error metadata: {meta_error}")
        
        raise
    
    finally:
        sot_client.close()


def main():
    parser = argparse.ArgumentParser(description="Download batch data from SOT ClickHouse")
    parser.add_argument("--processing-date", required=True, help="Processing date (YYYY-MM-DD)")
    parser.add_argument("--network", default="ethereum", help="Network (ethereum, bitcoin, polygon)")
    parser.add_argument("--sot-host", default="sot.clickhouse.example.com", help="SOT ClickHouse host")
    parser.add_argument("--sot-port", type=int, default=8123, help="SOT ClickHouse port")
    parser.add_argument("--sot-database", default="analyzers", help="SOT database name")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded without downloading")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")
    
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.processing_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {args.processing_date}. Expected YYYY-MM-DD")
        return 1
    
    try:
        download_from_sot(
            processing_date=args.processing_date,
            network=args.network,
            sot_host=args.sot_host,
            sot_port=args.sot_port,
            sot_database=args.sot_database,
            dry_run=args.dry_run,
            force=args.force
        )
        
        if not args.dry_run:
            logger.info("\nNext steps:")
            logger.info(f"1. Process batch: python scripts/process_batch.py --processing-date {args.processing_date} --network {args.network}")
            logger.info("2. Start API: python -m alert_scoring.api.server")
        
        return 0
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())