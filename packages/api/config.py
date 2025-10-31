from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_version: str = "1.0.0"
    
    cors_origins: List[str] = ["*"]
    
    github_url: str = "https://github.com/your-org/risk-scoring"
    commit_hash: str = "unknown"
    
    class Config:
        env_file = ".env"


settings = Settings()