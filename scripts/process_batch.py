import argparse
from datetime import datetime
import time
import pandas as pd
from loguru import logger

# TODO: Update when model files are created
# from alert_scoring.assessment.models.alert_scorer import AlertScorerModel
# from alert_scoring.assessment.models.alert_ranker import AlertRankerModel
# from alert_scoring.assessment.models.cluster_scorer import ClusterScorerModel
from alert_scoring.models.alert_scorer import AlertScorerModel
from alert_scoring.models.alert_ranker import AlertRankerModel
from alert_scoring.models.cluster_scorer import ClusterScorerModel
from alert_scoring.storage import get_connection_params, ClientFactory
from alert_scoring.storage.repositories import (
    AlertsRepository,
    FeaturesRepository,
    ClustersRepository,
    ScoresRepository,
    RankingsRepository,
    MetadataRepository
)


def pydantic_list_to_dataframe(pydantic_list):
    if not pydantic_list:
        return pd.DataFrame()
    
    return pd.DataFrame([item.model_dump() for item in pydantic_list])


def validate_scores(scores_df: pd.DataFrame):
    if scores_df['score'].min() < 0 or scores_df['score'].max() > 1:
        raise ValueError(f"Scores out of valid range [0, 1]: min={scores_df['score'].min()}, max={scores_df['score'].max()}")
    
    logger.info(f"Score validation passed: range [{scores_df['score'].min():.4f}, {scores_df['score'].max():.4f}]")


def load_input_data(network: str, processing_date: str, connection_params: dict) -> dict:
    logger.info(f"Loading input data for {processing_date}/{network} from ClickHouse")
    
    with ClientFactory(connection_params).client_context() as client:
        alerts_repo = AlertsRepository(client)
        features_repo = FeaturesRepository(client)
        clusters_repo = ClustersRepository(client)
        
        alerts_list = alerts_repo.get_alerts(processing_date, network)
        if not alerts_list:
            raise ValueError(f"No alerts found for {processing_date}/{network}")
        
        features_list = features_repo.get_features(processing_date, network)
        if not features_list:
            raise ValueError(f"No features found for {processing_date}/{network}")
        
        clusters_list = clusters_repo.get_clusters(processing_date, network)
        
        alerts_df = pydantic_list_to_dataframe(alerts_list)
        features_df = pydantic_list_to_dataframe(features_list)
        clusters_df = pydantic_list_to_dataframe(clusters_list)
        
        logger.info(f"Loaded {len(alerts_df)} alerts, {len(features_df)} features, {len(clusters_df)} clusters")
        
        return {
            'alerts': alerts_df,
            'features': features_df,
            'clusters': clusters_df
        }


def process_batch(network: str, processing_date: str,
                  alert_scorer_path: str, alert_ranker_path: str, cluster_scorer_path: str):
    start_time = time.time()
    
    logger.info(f"Processing batch for {processing_date}/{network}")
    
    connection_params = get_connection_params(network)
    
    try:
        data = load_input_data(network, processing_date, connection_params)
        alerts_df = data['alerts']
        features_df = data['features']
        clusters_df = data['clusters']
        
        logger.info("Loading models...")
        alert_scorer = AlertScorerModel()
        alert_scorer.load_model(alert_scorer_path)
        
        alert_ranker = AlertRankerModel()
        alert_ranker.load_model(alert_ranker_path)
        
        cluster_scorer = None
        if len(clusters_df) > 0:
            cluster_scorer = ClusterScorerModel()
            cluster_scorer.load_model(cluster_scorer_path)
        
        logger.info("Scoring alerts...")
        t0 = time.time()
        X_alerts = alert_scorer.prepare_features(alerts_df, features_df, clusters_df)
        alert_scores = alert_scorer.predict(X_alerts)
        alert_explanations = alert_scorer.create_explanations(X_alerts, alert_scores)
        alert_latency_ms = int((time.time() - t0) * 1000)
        alert_latency_per_item = alert_latency_ms // max(1, len(alerts_df))
        
        alert_scores_df = pd.DataFrame({
            'alert_id': alerts_df['alert_id'],
            'score': alert_scores,
            'model_version': alert_scorer.model_version,
            'latency_ms': alert_latency_per_item,
            'explain_json': alert_explanations
        })
        
        validate_scores(alert_scores_df)
        
        logger.info(f"Scored {len(alert_scores_df)} alerts in {alert_latency_ms}ms")
        
        logger.info("Ranking alerts...")
        t0 = time.time()
        X_rank = alert_ranker.prepare_features(alerts_df, features_df, clusters_df)
        alert_ranks = alert_ranker.predict(X_rank)
        rank_latency_ms = int((time.time() - t0) * 1000)
        
        alert_rankings_df = pd.DataFrame({
            'alert_id': alerts_df['alert_id'],
            'rank': alert_ranks,
            'model_version': alert_ranker.model_version
        })
        alert_rankings_df = alert_rankings_df.sort_values('rank')
        
        logger.info(f"Ranked {len(alert_rankings_df)} alerts in {rank_latency_ms}ms")
        
        cluster_scores_df = pd.DataFrame()
        cluster_latency_ms = 0
        if cluster_scorer and len(clusters_df) > 0:
            logger.info("Scoring clusters...")
            t0 = time.time()
            X_clusters = cluster_scorer.prepare_features(clusters_df, alerts_df, features_df)
            cluster_scores = cluster_scorer.predict(X_clusters)
            cluster_latency_ms = int((time.time() - t0) * 1000)
            
            cluster_scores_df = pd.DataFrame({
                'cluster_id': clusters_df['cluster_id'],
                'score': cluster_scores,
                'model_version': cluster_scorer.model_version
            })
            
            logger.info(f"Scored {len(cluster_scores_df)} clusters in {cluster_latency_ms}ms")
        
        logger.info(f"Writing results to ClickHouse for {processing_date}/{network}")
        
        with ClientFactory(connection_params).client_context() as client:
            scores_repo = ScoresRepository(client)
            rankings_repo = RankingsRepository(client)
            metadata_repo = MetadataRepository(client)
            
            scores_repo.insert_scores(alert_scores_df, processing_date, network)
            rankings_repo.insert_rankings(alert_rankings_df, processing_date, network)
            
            total_time = time.time() - start_time
            
            metadata = {
                'processed_at': datetime.utcnow(),
                'input_counts': {
                    'alerts': len(alerts_df),
                    'features': len(features_df),
                    'clusters': len(clusters_df)
                },
                'output_counts': {
                    'alert_scores': len(alert_scores_df),
                    'alert_rankings': len(alert_rankings_df),
                    'cluster_scores': len(cluster_scores_df)
                },
                'latencies_ms': {
                    'alert_scoring': alert_latency_ms,
                    'alert_ranking': rank_latency_ms,
                    'cluster_scoring': cluster_latency_ms,
                    'total': int(total_time * 1000)
                },
                'model_versions': {
                    'alert_scorer': alert_scorer.model_version,
                    'alert_ranker': alert_ranker.model_version,
                    'cluster_scorer': cluster_scorer.model_version if cluster_scorer else ''
                },
                'status': 'SUCCESS'
            }
            
            metadata_repo.insert_metadata(processing_date, network, metadata)
        
        logger.info(f"Batch processing completed in {total_time:.2f}s")
        logger.info(f"Results saved to ClickHouse for {processing_date}/{network}")
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        
        try:
            with ClientFactory(connection_params).client_context() as client:
                metadata_repo = MetadataRepository(client)
                
                error_metadata = {
                    'processed_at': datetime.utcnow(),
                    'input_counts': {
                        'alerts': 0,
                        'features': 0,
                        'clusters': 0
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
                    'status': 'FAILED',
                    'error_message': str(e)
                }
                
                metadata_repo.insert_metadata(processing_date, network, error_metadata)
                
        except Exception as meta_error:
            logger.error(f"Failed to write error metadata: {meta_error}")
        
        raise


def main():
    parser = argparse.ArgumentParser(description='Process batch of alerts for a specific date')
    parser.add_argument('--processing-date', required=True, help='Processing date (YYYY-MM-DD)')
    parser.add_argument('--network', default='ethereum', help='Network name (e.g., ethereum, bitcoin, polygon)')
    parser.add_argument('--alert-scorer', default='trained_models/alert_scorer_v1.0.0.txt', 
                       help='Path to alert scorer model')
    parser.add_argument('--alert-ranker', default='trained_models/alert_ranker_v1.0.0.txt',
                       help='Path to alert ranker model')
    parser.add_argument('--cluster-scorer', default='trained_models/cluster_scorer_v1.0.0.txt',
                       help='Path to cluster scorer model')
    
    args = parser.parse_args()
    
    try:
        datetime.strptime(args.processing_date, '%Y-%m-%d')
    except ValueError:
        logger.error(f"Invalid date format: {args.processing_date}. Expected YYYY-MM-DD")
        return 1
    
    try:
        metadata = process_batch(
            network=args.network,
            processing_date=args.processing_date,
            alert_scorer_path=args.alert_scorer,
            alert_ranker_path=args.alert_ranker,
            cluster_scorer_path=args.cluster_scorer
        )
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())