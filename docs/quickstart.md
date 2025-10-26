# Quick Start Guide

Get your AML Miner Template up and running in 5 minutes.

## Prerequisites

Before you begin, ensure you have:

- **Python 3.13.5+** installed
- **pip** package manager
- **git** for version control
- **8GB+ RAM** recommended
- **Linux/macOS/Windows** with WSL

## Installation

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/aml-miner-template.git
cd aml-miner-template
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional for quickstart)
# Default settings work for local development
```

## Running the API Server

### Start the Server

```bash
# Start the FastAPI server
python -m aml_miner.api.server
```

Expected output:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

The API is now running at `http://localhost:8000`

## Testing the API

### Check Health Status

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0"
}
```

### Score an Alert

```bash
curl -X POST http://localhost:8000/score/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [
      {
        "alert_id": "test-001",
        "network": "bitcoin",
        "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "amount_usd": 50000.0,
        "transaction_count": 10,
        "risk_category": "high_value",
        "timestamp": "2025-01-15T10:30:00Z"
      }
    ]
  }'
```

Expected response:
```json
{
  "scores": [
    {
      "alert_id": "test-001",
      "score": 0.75,
      "model_version": "1.0.0"
    }
  ]
}
```

### Rank Alerts

```bash
curl -X POST http://localhost:8000/rank/alerts \
  -H "Content-Type: application/json" \
  -d '{
    "alerts": [
      {
        "alert_id": "alert-001",
        "network": "bitcoin",
        "address": "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
        "amount_usd": 50000.0,
        "transaction_count": 10,
        "risk_category": "high_value",
        "timestamp": "2025-01-15T10:30:00Z"
      },
      {
        "alert_id": "alert-002",
        "network": "ethereum",
        "address": "0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
        "amount_usd": 25000.0,
        "transaction_count": 5,
        "risk_category": "medium_value",
        "timestamp": "2025-01-15T11:00:00Z"
      }
    ]
  }'
```

Expected response:
```json
{
  "rankings": [
    {
      "alert_id": "alert-001",
      "rank": 1,
      "score": 0.85
    },
    {
      "alert_id": "alert-002",
      "rank": 2,
      "score": 0.62
    }
  ]
}
```

### Score a Cluster

```bash
curl -X POST http://localhost:8000/score/clusters \
  -H "Content-Type: application/json" \
  -d '{
    "clusters": [
      {
        "cluster_id": "cluster-001",
        "network": "bitcoin",
        "total_volume_usd": 1000000.0,
        "transaction_count": 150,
        "unique_addresses": 45,
        "pattern_matches": ["mixing", "layering"],
        "risk_level": "high"
      }
    ]
  }'
```

Expected response:
```json
{
  "scores": [
    {
      "cluster_id": "cluster-001",
      "score": 0.92,
      "model_version": "1.0.0"
    }
  ]
}
```

## Docker Deployment (Optional)

### Using Docker Compose

```bash
# Build and start containers
docker-compose up --build

# The API will be available at http://localhost:8000
```

### Using Docker Directly

```bash
# Build the image
docker build -t aml-miner-template .

# Run the container
docker run -p 8000:8000 aml-miner-template
```

## Next Steps

Now that your API is running, you can:

1. **Train Custom Models** - See [`training_guide.md`](training_guide.md)
2. **Customize Features** - See [`customization.md`](customization.md)
3. **Explore API** - See [`api_reference.md`](api_reference.md)
4. **Production Deployment** - See [`README.md`](../README.md)

## Troubleshooting

### Port Already in Use

If port 8000 is already in use:

```bash
# Run on a different port
python -m aml_miner.api.server --port 8001
```

Or set in `.env`:
```
API_PORT=8001
```

### Missing Dependencies

If you get import errors:

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Python Version Issues

Ensure Python 3.13.5 or higher:

```bash
python --version
# Should output: Python 3.13.5 or higher
```

### Model Not Found Errors

The template works with default models. If you see model loading errors:

```bash
# Check model configuration
cat aml_miner/config/model_config.yaml
```

Default configuration uses simple models that don't require training data.

### Connection Refused

Ensure the server is running and listening on the correct host:

```bash
# Check if server is running
curl http://localhost:8000/health

# If using Docker, ensure port mapping is correct
docker ps
```

### Memory Issues

If you encounter memory errors:

- Reduce batch size in requests
- Increase available RAM
- Use Docker with memory limits

### Database Connection Issues

The template doesn't require a database for basic operation. If you've added database features:

- Check database credentials in `.env`
- Ensure database server is running
- Verify network connectivity

## Development Mode

For development with auto-reload:

```bash
# Install development dependencies
pip install uvicorn[standard]

# Run with auto-reload
uvicorn aml_miner.api.server:app --reload --port 8000
```

## Verification

Run the verification script to ensure everything is working:

```bash
python scripts/verify_api.py
```

Expected output:
```
✓ Health check passed
✓ Alert scoring passed
✓ Alert ranking passed
✓ Cluster scoring passed
All API endpoints working correctly!
```

## Getting Help

- **Documentation**: Check [`docs/`](.) for detailed guides
- **Issues**: Open an issue on GitHub
- **API Reference**: See [`api_reference.md`](api_reference.md)

## What's Next?

You've successfully set up the AML Miner Template! Here are recommended next steps:

1. **Understand the architecture** - Read [`README.md`](../README.md)
2. **Train your models** - Follow [`training_guide.md`](training_guide.md)
3. **Customize features** - Learn from [`customization.md`](customization.md)
4. **Deploy to production** - Use Docker for production deployment
5. **Monitor performance** - Set up logging and monitoring

Happy mining! 🚀