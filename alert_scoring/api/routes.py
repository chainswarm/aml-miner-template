from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from loguru import logger

from alert_scoring.storage import ClientFactory, get_connection_params
from alert_scoring.storage.repositories.scores_repository import ScoresRepository
from alert_scoring.storage.repositories.rankings_repository import RankingsRepository, AlertRanking
from alert_scoring.storage.repositories.cluster_scores_repository import ClusterScoresRepository
from alert_scoring.storage.repositories.metadata_repository import MetadataRepository, BatchMetadata
from alert_scoring.api.schemas import ScoreResponse, MetadataResponse, ClusterScore
from alert_scoring.config.settings import Settings

router = APIRouter()
settings = Settings()


def get_client_factory(network: str = None) -> ClientFactory:
    if network is None:
        network = settings.NETWORK
    connection_params = get_connection_params(network)
    return ClientFactory(connection_params)


@router.get("/health")
def health_check():
    return {
        "status": "healthy",
        "storage": "clickhouse",
        "default_network": settings.NETWORK
    }


@router.get("/version")
def get_version():
    return {
        "api_version": "1.0.0",
        "storage_backend": "clickhouse"
    }


@router.get("/dates/available")
def get_available_dates(network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            scores_repo = ScoresRepository(client)
            dates = scores_repo.get_available_dates(network)
            
            if not dates:
                logger.warning(f"No dates available for network {network}")
                return []
            
            return dates
    
    except Exception as e:
        logger.error(f"Error retrieving available dates for {network}: {e}")
        raise HTTPException(500, f"Error retrieving dates: {e}")


@router.get("/dates/latest")
def get_latest_date(network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            scores_repo = ScoresRepository(client)
            latest = scores_repo.get_latest_date(network)
            
            if not latest:
                raise HTTPException(404, f"No processing dates available for network {network}")
            
            return {"processing_date": latest, "network": network}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest date for {network}: {e}")
        raise HTTPException(500, f"Error retrieving latest date: {e}")


@router.get("/scores/alerts/latest")
def get_latest_alert_scores(network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            scores_repo = ScoresRepository(client)
            
            latest_date = scores_repo.get_latest_date(network)
            
            if not latest_date:
                raise HTTPException(404, f"No scores available for network {network}")
            
            scores = scores_repo.get_scores(latest_date, network)
            
            return {
                "processing_date": latest_date,
                "network": network,
                "scores": [score.dict() for score in scores],
                "total_scores": len(scores)
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest scores for {network}: {e}")
        raise HTTPException(500, f"Error retrieving scores: {e}")


@router.get("/scores/alerts/{processing_date}", response_model=List[ScoreResponse])
def get_alert_scores(processing_date: str, network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            scores_repo = ScoresRepository(client)
            scores = scores_repo.get_scores(processing_date, network)
            
            if not scores:
                raise HTTPException(
                    404,
                    f"No scores found for date {processing_date} and network {network}"
                )
            
            return [score.dict() for score in scores]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving scores for {processing_date}/{network}: {e}")
        raise HTTPException(500, f"Error retrieving scores: {e}")


@router.get("/scores/rankings/latest", response_model=List[AlertRanking])
def get_latest_rankings(network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            rankings_repo = RankingsRepository(client)
            
            latest_date = rankings_repo.get_latest_date(network)
            
            if not latest_date:
                raise HTTPException(404, f"No rankings available for network {network}")
            
            rankings = rankings_repo.get_rankings(latest_date, network)
            
            return rankings
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest rankings for {network}: {e}")
        raise HTTPException(500, f"Error retrieving rankings: {e}")


@router.get("/scores/rankings/{processing_date}", response_model=List[AlertRanking])
def get_rankings_by_date(processing_date: str, network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            rankings_repo = RankingsRepository(client)
            rankings = rankings_repo.get_rankings(processing_date, network)
            
            if not rankings:
                raise HTTPException(
                    404,
                    f"No rankings found for date {processing_date} and network {network}"
                )
            
            return rankings
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving rankings for {processing_date}/{network}: {e}")
        raise HTTPException(500, f"Error retrieving rankings: {e}")


@router.get("/scores/rankings/top/{n}", response_model=List[AlertRanking])
def get_top_rankings(n: int, processing_date: Optional[str] = None, network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            rankings_repo = RankingsRepository(client)
            
            if processing_date is None:
                processing_date = rankings_repo.get_latest_date(network)
                if not processing_date:
                    raise HTTPException(404, f"No rankings available for network {network}")
            
            rankings = rankings_repo.get_top_n(n, processing_date, network)
            
            if not rankings:
                raise HTTPException(
                    404,
                    f"No rankings found for date {processing_date} and network {network}"
                )
            
            return rankings
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving top {n} rankings for {network}: {e}")
        raise HTTPException(500, f"Error retrieving top rankings: {e}")


@router.get("/scores/clusters/latest", response_model=List[ClusterScore])
def get_latest_cluster_scores(network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            cluster_scores_repo = ClusterScoresRepository(client)
            
            latest_date = cluster_scores_repo.get_latest_date(network)
            
            if not latest_date:
                raise HTTPException(404, f"No cluster scores available for network {network}")
            
            scores = cluster_scores_repo.get_scores(latest_date, network)
            
            return scores
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest cluster scores for {network}: {e}")
        raise HTTPException(500, f"Error retrieving cluster scores: {e}")


@router.get("/scores/clusters/{processing_date}", response_model=List[ClusterScore])
def get_cluster_scores_by_date(processing_date: str, network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            cluster_scores_repo = ClusterScoresRepository(client)
            scores = cluster_scores_repo.get_scores(processing_date, network)
            
            if not scores:
                raise HTTPException(
                    404,
                    f"No cluster scores found for date {processing_date} and network {network}"
                )
            
            return scores
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving cluster scores for {processing_date}/{network}: {e}")
        raise HTTPException(500, f"Error retrieving cluster scores: {e}")


@router.get("/scores/clusters/{cluster_id}/{processing_date}", response_model=ClusterScore)
def get_cluster_score_by_id(cluster_id: str, processing_date: str, network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            cluster_scores_repo = ClusterScoresRepository(client)
            score = cluster_scores_repo.get_score_by_id(cluster_id, processing_date, network)
            
            if not score:
                raise HTTPException(
                    404,
                    f"No cluster score found for cluster_id {cluster_id}, date {processing_date}, network {network}"
                )
            
            return score
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving cluster score for {cluster_id}/{processing_date}/{network}: {e}")
        raise HTTPException(500, f"Error retrieving cluster score: {e}")


@router.get("/metadata/latest", response_model=BatchMetadata)
def get_latest_metadata(network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            metadata_repo = MetadataRepository(client)
            metadata = metadata_repo.get_latest_metadata(network)
            
            if not metadata:
                raise HTTPException(404, f"No batch metadata available for network {network}")
            
            return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest metadata for {network}: {e}")
        raise HTTPException(500, f"Error retrieving metadata: {e}")


@router.get("/metadata/{processing_date}", response_model=BatchMetadata)
def get_metadata_by_date(processing_date: str, network: str = Query(settings.NETWORK)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            metadata_repo = MetadataRepository(client)
            metadata = metadata_repo.get_metadata(processing_date, network)
            
            if not metadata:
                raise HTTPException(
                    404,
                    f"No batch metadata found for date {processing_date} and network {network}"
                )
            
            return metadata
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metadata for {processing_date}/{network}: {e}")
        raise HTTPException(500, f"Error retrieving metadata: {e}")


@router.get("/metadata/history", response_model=List[BatchMetadata])
def get_metadata_history(network: str = Query(settings.NETWORK), limit: int = Query(30, ge=1, le=365)):
    client_factory = get_client_factory(network)
    
    try:
        with client_factory.client_context() as client:
            query = f'''
                SELECT processing_date, network, processed_at,
                       input_counts_alerts, input_counts_features, input_counts_clusters,
                       output_counts_alert_scores, output_counts_alert_rankings, output_counts_cluster_scores,
                       latencies_ms_alert_scoring, latencies_ms_alert_ranking, latencies_ms_cluster_scoring, latencies_ms_total,
                       model_versions_alert_scorer, model_versions_alert_ranker, model_versions_cluster_scorer,
                       status, error_message
                FROM batch_metadata
                WHERE network = %(network)s
                ORDER BY processing_date DESC
                LIMIT %(limit)s
            '''
            
            result = client.query(query, {'network': network, 'limit': limit})
            
            from alert_scoring.storage.utils import rows_to_pydantic_list
            metadata_list = rows_to_pydantic_list(
                BatchMetadata,
                result.result_rows,
                result.column_names
            )
            
            if not metadata_list:
                raise HTTPException(404, f"No batch metadata history available for network {network}")
            
            return metadata_list
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving metadata history for {network}: {e}")
        raise HTTPException(500, f"Error retrieving metadata history: {e}")


@router.post("/refresh")
def refresh_database(network: str = Query(settings.NETWORK)):
    return {
        "status": "success",
        "message": "ClickHouse doesn't require refresh - data is immediately available",
        "network": network
    }