from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_connected: bool


class VersionResponse(BaseModel):
    api_version: str
    models: Dict[str, str]
    network: str
    github_url: str
    commit_hash: str


class NetworkInfo(BaseModel):
    network: str
    status: str
    latest_processing_date: Optional[str] = None
    model_versions: Optional[Dict[str, str]] = None
    total_alerts_scored: Optional[int] = None
    last_training_date: Optional[str] = None


class NetworksResponse(BaseModel):
    networks: List[NetworkInfo]
    total_networks: int
    miner_type: str


class NetworkStatistics(BaseModel):
    total_alerts_scored: int
    total_clusters_scored: int
    avg_alerts_per_day: int
    first_date: str
    last_date: str


class ModelVersionInfo(BaseModel):
    version: str
    trained_at: datetime
    performance: Optional[Dict[str, float]] = None


class NetworkDetailResponse(BaseModel):
    network: str
    status: str
    latest_processing_date: str
    available_dates: List[str]
    total_dates: int
    model_versions: Dict[str, ModelVersionInfo]
    statistics: NetworkStatistics
    last_training_date: str
    next_training_date: str


class DatesAvailableResponse(BaseModel):
    dates: List[str]
    total_count: int


class LatestDateResponse(BaseModel):
    latest_date: str
    processed_at: datetime


class AlertScore(BaseModel):
    alert_id: str
    score: float = Field(..., ge=0.0, le=1.0)


class AlertScoresMetadata(BaseModel):
    processed_at: datetime
    total_latency_ms: int
    avg_latency_ms: float


class AlertScoresResponse(BaseModel):
    processing_date: str
    model_version: str
    total_alerts: int
    scores: List[AlertScore]
    metadata: AlertScoresMetadata


class AlertRanking(BaseModel):
    alert_id: str
    rank: int
    score: float = Field(..., ge=0.0, le=1.0)


class AlertRankingsMetadata(BaseModel):
    processed_at: datetime


class AlertRankingsResponse(BaseModel):
    processing_date: str
    model_version: str
    total_alerts: int
    rankings: List[AlertRanking]
    metadata: AlertRankingsMetadata


class ClusterScore(BaseModel):
    cluster_id: str
    score: float = Field(..., ge=0.0, le=1.0)


class ClusterScoresMetadata(BaseModel):
    processed_at: datetime
    total_latency_ms: int
    avg_latency_ms: float


class ClusterScoresResponse(BaseModel):
    processing_date: str
    model_version: str
    total_clusters: int
    scores: List[ClusterScore]
    metadata: ClusterScoresMetadata


class ModelHyperparameters(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    objective: str
    random_state: int = 42
    n_jobs: int = -1
    tree_method: str = "hist"


class TrainingData(BaseModel):
    start_date: str
    end_date: str
    window_days: int
    num_samples: int
    num_positive: int
    num_negative: int
    positive_rate: float
    num_features: int
    feature_names: Optional[List[str]] = None


class ModelPerformance(BaseModel):
    train_auc: float
    val_auc: float
    train_logloss: float
    val_logloss: float
    train_samples: Optional[int] = None
    val_samples: Optional[int] = None


class FeatureImportance(BaseModel):
    feature: str
    importance: float
    rank: int


class ModelMetadata(BaseModel):
    model_type: str
    model_version: str
    trained_at: datetime
    training_duration_minutes: int
    algorithm: str
    hyperparameters: ModelHyperparameters
    training_data: TrainingData
    performance: ModelPerformance
    feature_importance: List[FeatureImportance]
    model_size_mb: float
    model_path: str
    download_url: str
    github_url: str
    commit_hash: str


class AllModelsMetadataResponse(BaseModel):
    models: List[ModelMetadata]
    github_url: str
    commit_hash: str