# Miner API

FastAPI server that exposes pre-computed risk scores to validators.

## Quick Start

```bash
# Start the API server
python scripts/start_api.py

# With custom port
python scripts/start_api.py --port 8001

# With auto-reload for development
python scripts/start_api.py --reload
```

## Configuration

Set environment variables in `.env`:

```bash
# ClickHouse (per network via get_connection_params)
CLICKHOUSE_HOST=localhost
CLICKHOUSE_PORT=8123
CLICKHOUSE_USER=default
CLICKHOUSE_PASSWORD=miner

# API
API_HOST=0.0.0.0
API_PORT=8000

# Repository
GITHUB_URL=https://github.com/your-org/risk-scoring
COMMIT_HASH=$(git rev-parse HEAD)
```

## Network Support

Network is **always** a required query parameter. The API dynamically creates ClickHouse connections based on the network parameter using `get_connection_params(network)`.

Example networks: `torus`, `bitcoin`, `ethereum`

## Endpoints

All endpoints require `network` query parameter.

### Info
- `GET /health` - Health check (no network required)
- `GET /version?network=torus` - API and model versions

### Dates
- `GET /dates/available?network=torus` - Available processing dates
- `GET /dates/latest?network=torus` - Latest processing date

### Scores
- `GET /scores/alerts/latest?network=torus` - Latest alert scores
- `GET /scores/alerts/{processing_date}?network=torus` - Alert scores for date
- `GET /rankings/alerts/latest?network=torus` - Latest alert rankings  
- `GET /rankings/alerts/{processing_date}?network=torus` - Alert rankings for date
- `GET /scores/clusters/latest?network=torus` - Latest cluster scores
- `GET /scores/clusters/{processing_date}?network=torus` - Cluster scores for date

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Get latest scores for torus network
curl "http://localhost:8000/scores/alerts/latest?network=torus"

# Get scores for specific date
curl "http://localhost:8000/scores/alerts/2025-10-30?network=torus"

# Test different network
curl "http://localhost:8000/scores/alerts/latest?network=bitcoin"
```

## API Documentation

Interactive API docs available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## How It Works

1. Network parameter in query determines which database to connect to
2. `get_connection_params(network)` returns connection config for that network
3. `ClientFactory` creates connection to the appropriate ClickHouse database
4. API serves pre-computed scores from that database