from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from packages.api.routes import router
from packages.api.config import settings

app = FastAPI(
    title="Miner API",
    version=settings.api_version,
    description="Risk scoring miner API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(router)