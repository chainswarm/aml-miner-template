from fastapi import APIRouter, HTTPException, Path, Query
from datetime import datetime

from packages.api.models import *
from packages.api import database as db
from packages.api.config import settings
from packages.storage import get_connection_params, ClientFactory

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version=settings.api_version,
        database_connected=True
    )


@router.get("/version", response_model=VersionResponse)
async def get_version(network: str = Query(..., description="Network identifier")):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        latest_date = db.get_latest_date(client)
        if not latest_date:
            raise HTTPException(404, f"No data available for network: {network}")
        
        metadata = db.get_batch_metadata(client, latest_date)
        if not metadata:
            raise HTTPException(404, f"No metadata available for network: {network}")
        
        return VersionResponse(
            api_version=settings.api_version,
            models={
                "alert_scorer": metadata.get('model_versions_alert_scorer', 'unknown'),
                "alert_ranker": metadata.get('model_versions_alert_ranker', 'unknown'),
                "cluster_scorer": metadata.get('model_versions_cluster_scorer', 'unknown')
            },
            network=network,
            github_url=settings.github_url,
            commit_hash=settings.commit_hash
        )


@router.get("/dates/available", response_model=DatesAvailableResponse)
async def get_available_dates(network: str = Query(..., description="Network identifier")):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        dates = db.get_available_dates(client)
        return DatesAvailableResponse(dates=dates, total_count=len(dates))


@router.get("/dates/latest", response_model=LatestDateResponse)
async def get_latest_date(network: str = Query(..., description="Network identifier")):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        latest_date = db.get_latest_date(client)
        if not latest_date:
            raise HTTPException(404, f"No data available for network: {network}")
        
        metadata = db.get_batch_metadata(client, latest_date)
        if not metadata:
            raise HTTPException(404, f"No metadata available for date: {latest_date}")
        
        return LatestDateResponse(
            latest_date=latest_date,
            processed_at=metadata.get('created_at', datetime.utcnow())
        )


@router.get("/scores/alerts/latest", response_model=AlertScoresResponse)
async def get_latest_alert_scores(network: str = Query(..., description="Network identifier")):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        latest_date = db.get_latest_date(client)
        if not latest_date:
            raise HTTPException(404, f"No data available for network: {network}")
        return _get_alert_scores_impl(client, latest_date)


@router.get("/scores/alerts/{processing_date}", response_model=AlertScoresResponse)
async def get_alert_scores(
    processing_date: str = Path(..., description="Processing date (YYYY-MM-DD)"),
    network: str = Query(..., description="Network identifier")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        return _get_alert_scores_impl(client, processing_date)


def _get_alert_scores_impl(client, processing_date: str) -> AlertScoresResponse:
    scores = db.get_alert_scores(client, processing_date)
    metadata = db.get_batch_metadata(client, processing_date)
    
    if not metadata:
        return AlertScoresResponse(
            processing_date=processing_date,
            model_version='unknown',
            total_alerts=0,
            scores=[],
            metadata=AlertScoresMetadata(
                processed_at=datetime.utcnow(),
                total_latency_ms=0,
                avg_latency_ms=0
            )
        )
    
    total_latency = metadata.get('latencies_ms_total', 0)
    avg_latency = total_latency / len(scores) if scores else 0
    
    return AlertScoresResponse(
        processing_date=processing_date,
        model_version=metadata.get('model_versions_alert_scorer', 'unknown'),
        total_alerts=len(scores),
        scores=[AlertScore(**s) for s in scores],
        metadata=AlertScoresMetadata(
            processed_at=metadata.get('created_at', datetime.utcnow()),
            total_latency_ms=total_latency,
            avg_latency_ms=avg_latency
        )
    )


@router.get("/rankings/alerts/latest", response_model=AlertRankingsResponse)
async def get_latest_alert_rankings(network: str = Query(..., description="Network identifier")):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        latest_date = db.get_latest_date(client)
        if not latest_date:
            raise HTTPException(404, f"No data available for network: {network}")
        return _get_alert_rankings_impl(client, latest_date)


@router.get("/rankings/alerts/{processing_date}", response_model=AlertRankingsResponse)
async def get_alert_rankings(
    processing_date: str = Path(..., description="Processing date (YYYY-MM-DD)"),
    network: str = Query(..., description="Network identifier")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        return _get_alert_rankings_impl(client, processing_date)


def _get_alert_rankings_impl(client, processing_date: str) -> AlertRankingsResponse:
    rankings = db.get_alert_rankings(client, processing_date)
    metadata = db.get_batch_metadata(client, processing_date)
    
    if not metadata:
        return AlertRankingsResponse(
            processing_date=processing_date,
            model_version='unknown',
            total_alerts=0,
            rankings=[],
            metadata=AlertRankingsMetadata(processed_at=datetime.utcnow())
        )
    
    return AlertRankingsResponse(
        processing_date=processing_date,
        model_version=metadata.get('model_versions_alert_ranker', 'unknown'),
        total_alerts=len(rankings),
        rankings=[AlertRanking(**r) for r in rankings],
        metadata=AlertRankingsMetadata(processed_at=metadata.get('created_at', datetime.utcnow()))
    )


@router.get("/scores/clusters/latest", response_model=ClusterScoresResponse)
async def get_latest_cluster_scores(network: str = Query(..., description="Network identifier")):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        latest_date = db.get_latest_date(client)
        if not latest_date:
            raise HTTPException(404, f"No data available for network: {network}")
        return _get_cluster_scores_impl(client, latest_date)


@router.get("/scores/clusters/{processing_date}", response_model=ClusterScoresResponse)
async def get_cluster_scores(
    processing_date: str = Path(..., description="Processing date (YYYY-MM-DD)"),
    network: str = Query(..., description="Network identifier")
):
    connection_params = get_connection_params(network)
    factory = ClientFactory(connection_params)
    
    with factory.client_context() as client:
        return _get_cluster_scores_impl(client, processing_date)


def _get_cluster_scores_impl(client, processing_date: str) -> ClusterScoresResponse:
    scores = db.get_cluster_scores(client, processing_date)
    metadata = db.get_batch_metadata(client, processing_date)
    
    if not metadata:
        return ClusterScoresResponse(
            processing_date=processing_date,
            model_version='unknown',
            total_clusters=0,
            scores=[],
            metadata=ClusterScoresMetadata(
                processed_at=datetime.utcnow(),
                total_latency_ms=0,
                avg_latency_ms=0
            )
        )
    
    total_latency = metadata.get('latencies_ms_cluster_scoring', 0)
    avg_latency = total_latency / len(scores) if scores else 0
    
    return ClusterScoresResponse(
        processing_date=processing_date,
        model_version=metadata.get('model_versions_cluster_scorer', 'unknown'),
        total_clusters=len(scores),
        scores=[ClusterScore(**s) for s in scores],
        metadata=ClusterScoresMetadata(
            processed_at=metadata.get('created_at', datetime.utcnow()),
            total_latency_ms=total_latency,
            avg_latency_ms=avg_latency
        )
    )