# Docker Deployment Guide

## Overview

This guide explains how to build and run the Spotify Recommendation Engine using Docker and Docker Compose.

## Prerequisites

- Docker >= 20.10
- Docker Compose >= 1.29
- 2GB+ available disk space

## Quick Start

### Option 1: Using Docker Compose (Recommended)

Start the entire stack (Streamlit + MLflow) with one command:

```bash
docker-compose up -d
```

This will:
- Build the Streamlit app image
- Start Streamlit on http://localhost:8501
- Start MLflow tracking server on http://localhost:5000
- Mount `data/` and `models/` as volumes for persistence

### Option 2: Using Docker Only

Build and run just the Streamlit app:

```bash
# Build the image
docker build -t spotify-recsys:latest .

# Run the container
docker run -d \
  --name spotify-recsys \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  spotify-recsys:latest
```

## Accessing the Applications

### Streamlit App
```
http://localhost:8501
```

### MLflow Tracking Server (if using docker-compose)
```
http://localhost:5000
```

## Container Management

### View logs
```bash
# Docker Compose
docker-compose logs -f spotify-recsys

# Docker only
docker logs -f spotify-recsys
```

### Stop containers
```bash
# Docker Compose
docker-compose down

# Docker only
docker stop spotify-recsys
docker rm spotify-recsys
```

### Check container status
```bash
docker-compose ps
```

## Volumes

Data persistence is handled through Docker volumes:

- **data/**: Feature-engineered data and datasets
- **models/**: Trained model files (pkl, json)
- **results/**: Model comparison and tuning results
- **notebooks/**: Jupyter notebooks

These directories are mounted from your host machine, so changes persist after container restarts.

## Environment Variables

Configure the app using `.env` file:

```bash
cp .env.example .env
# Edit .env with your settings
docker-compose up
```

Common settings:
- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Bind address (default: 0.0.0.0)
- `STREAMLIT_LOGGER_LEVEL`: Log verbosity (default: info)

## Health Checks

Both services have health checks configured:

```bash
# Check health
docker-compose ps

# View health status
docker ps --format "table {{.Names}}\t{{.Status}}"
```

## Building and Pushing to Registry

### Build a custom image
```bash
docker build -t spotify-recsys:v1.0 .
```

### Push to Docker Hub
```bash
docker tag spotify-recsys:v1.0 <your-username>/spotify-recsys:v1.0
docker push <your-username>/spotify-recsys:v1.0
```

### Push to other registries (e.g., AWS ECR)
```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag spotify-recsys:latest <account>.dkr.ecr.us-east-1.amazonaws.com/spotify-recsys:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/spotify-recsys:latest
```

## Scaling (with Docker Compose)

For production, modify `docker-compose.yml` to scale:

```bash
docker-compose up -d --scale mlflow-server=1
```

## Troubleshooting

### Port already in use
```bash
# Free up port 8501
lsof -ti:8501 | xargs kill -9

# Or use different port
docker run -p 8502:8501 spotify-recsys:latest
```

### Container exits immediately
```bash
# Check logs
docker logs spotify-recsys

# Common issues:
# - Missing dependencies in requirements.txt
# - Port conflicts
# - Insufficient disk space
```

### Data not persisting
```bash
# Verify volumes are mounted
docker inspect spotify-recsys | grep -A 10 Mounts

# Check volume permissions
ls -la data/ models/
```

## Production Deployment

For production use (e.g., AWS, GCP, Azure):

1. **Push to container registry:**
   ```bash
   docker push <registry>/spotify-recsys:latest
   ```

2. **Deploy to orchestration platform:**
   - **AWS ECS**: Create task definition with the image
   - **Kubernetes**: Use the provided image in deployment manifest
   - **Heroku**: Deploy using Container Registry
   - **Cloud Run (GCP)**: Deploy from container image

3. **Environment configuration:**
   - Use secrets manager for sensitive data
   - Configure environment variables in the orchestration platform
   - Mount persistent storage for data and models

## Docker Image Size Optimization

The image uses `python:3.9-slim` to minimize size (~500MB with dependencies).

To further optimize:
- Remove unused packages from `requirements.txt`
- Use multi-stage builds for production

## Next Steps

1. Verify the app loads at http://localhost:8501
2. Check MLflow at http://localhost:5000
3. Verify model predictions work
4. Deploy to cloud platform of choice

For more information, see the main [README.md](README.md)
