from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    TRAINED_MODELS_DIR: Path = BASE_DIR / "trained_models"
    DATA_DIR: Path = BASE_DIR / "data"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    INPUT_DIR: Path = BASE_DIR / "input"
    OUTPUT_DIR: Path = BASE_DIR / "output"
    TRAINING_DATA_DIR: Path = BASE_DIR / "training_data"
    VALIDATION_RESULTS_DIR: Path = BASE_DIR / "validation_results"
    
    ALERT_SCORER_PATH: Path = TRAINED_MODELS_DIR / "alert_scorer_v1.0.0.txt"
    ALERT_RANKER_PATH: Path = TRAINED_MODELS_DIR / "alert_ranker_v1.0.0.txt"
    CLUSTER_SCORER_PATH: Path = TRAINED_MODELS_DIR / "cluster_scorer_v1.0.0.txt"
    
    CLICKHOUSE_HOST: str = "localhost"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_DATABASE: str = "alert_scoring"
    CLICKHOUSE_USER: str = "default"
    CLICKHOUSE_PASSWORD: str = ""
    
    NETWORK: str = "ethereum"
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_TIMEOUT: int = 120
    
    MAX_FEATURES: int = 100
    FEATURE_SELECTION_METHOD: str = "importance"
    
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Path = BASE_DIR / "logs" / "api.log"
    
    RANDOM_SEED: int = 42
    
    BATCH_SIZE: int = 1000
    MAX_MEMORY_MB: int = 2048
    
    RETENTION_DAYS: int = 90
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"