from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import uvicorn
import sys
import time

from aml_miner.config import Settings
from aml_miner.models import AlertScorerModel, AlertRankerModel, ClusterScorerModel
from aml_miner.utils.determinism import set_deterministic_mode
from aml_miner.version import __version__
from aml_miner.api import routes

app = FastAPI(
    title="AML Miner API",
    description="Alert scoring, ranking, and cluster assessment for AML detection",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

alert_scorer = None
alert_ranker = None
cluster_scorer = None
settings = None


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    logger.info(f"Request: {request.method} {request.url.path}")
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    logger.info(f"Response: {request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")
    
    response.headers["X-Process-Time"] = str(process_time)
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


@app.on_event("startup")
async def startup_event():
    global alert_scorer, alert_ranker, cluster_scorer, settings
    
    logger.info("Starting AML Miner API...")
    logger.info(f"API Version: {__version__}")
    
    try:
        settings = Settings()
        logger.info(f"Loaded settings from: {settings.BASE_DIR}")
        
        set_deterministic_mode(settings.RANDOM_SEED)
        logger.info(f"Set deterministic mode with seed: {settings.RANDOM_SEED}")
        
        settings.TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        alert_scorer = AlertScorerModel()
        if settings.ALERT_SCORER_PATH.exists():
            alert_scorer.load_model(settings.ALERT_SCORER_PATH)
            logger.info(f" Loaded alert scorer from: {settings.ALERT_SCORER_PATH}")
        else:
            logger.warning(f"Alert scorer model not found at: {settings.ALERT_SCORER_PATH}")
        
        alert_ranker = AlertRankerModel()
        if settings.ALERT_RANKER_PATH.exists():
            alert_ranker.load_model(settings.ALERT_RANKER_PATH)
            logger.info(f" Loaded alert ranker from: {settings.ALERT_RANKER_PATH}")
        else:
            logger.warning(f"Alert ranker model not found at: {settings.ALERT_RANKER_PATH}")
        
        cluster_scorer = ClusterScorerModel()
        if settings.CLUSTER_SCORER_PATH.exists():
            cluster_scorer.load_model(settings.CLUSTER_SCORER_PATH)
            logger.info(f" Loaded cluster scorer from: {settings.CLUSTER_SCORER_PATH}")
        else:
            logger.warning(f"Cluster scorer model not found at: {settings.CLUSTER_SCORER_PATH}")
        
        routes.set_models(alert_scorer, alert_ranker, cluster_scorer, settings)
        
        logger.info(" API started successfully")
        logger.info(f"API listening on {settings.API_HOST}:{settings.API_PORT}")
        
    except Exception as e:
        logger.error(f"Failed to start API: {str(e)}", exc_info=True)
        sys.exit(1)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down AML Miner API...")
    
    try:
        global alert_scorer, alert_ranker, cluster_scorer
        
        alert_scorer = None
        alert_ranker = None
        cluster_scorer = None
        
        logger.info(" API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


app.include_router(routes.router)


@app.get("/")
async def root():
    return {
        "name": "AML Miner API",
        "version": __version__,
        "status": "running",
        "endpoints": {
            "health": "/health",
            "version": "/version",
            "score_alerts": "/score/alerts",
            "rank_alerts": "/rank/alerts",
            "score_clusters": "/score/clusters",
            "docs": "/docs"
        }
    }


def main():
    settings = Settings()
    
    logger.add(
        settings.LOG_FILE,
        rotation="500 MB",
        retention="10 days",
        level=settings.LOG_LEVEL
    )
    
    logger.info(f"Starting AML Miner API server v{__version__}")
    
    uvicorn.run(
        "aml_miner.api.server:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False
    )


if __name__ == "__main__":
    main()