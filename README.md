# State Forge API - Railway Deployment

## Overview
FastAPI ML backend for Automata Theory Conversions with 4 PyTorch models for DFA minimization, regex-to-ε-NFA, ε-NFA-to-DFA, and PDA conversions.

## Railway Deployment (Docker)

### Required Environment Variables
Set these in your Railway dashboard:

**Optional Configuration:**
- `DEBUG` - Set to "false" for production (default: "true")
- `ENVIRONMENT` - Set to "production" for production (default: "development")
- `ALLOWED_ORIGINS` - Comma-separated list of allowed origins for CORS (default: "*")

**Automatically Set by Railway:**
- `PORT` - Railway sets this automatically
- `HOST` - Default is "0.0.0.0"

### Deployment Files
- `Dockerfile` - Multi-stage Docker build optimized for production
- `Procfile` - Gunicorn server configuration (backup)
- `railway.toml` - Railway deployment settings configured for Docker
- `.railwayignore` - Files to exclude from deployment
- `.dockerignore` - Files to exclude from Docker build

### API Endpoints
- `GET /` - Health check
- `POST /api/v1/convert` - Model conversion endpoint
- `GET /api/v1/models` - List available models

### Models Included
- DFA Minimization (~7.2MB)
- Regex-to-ε-NFA
- ε-NFA-to-DFA  
- PDA (Push Down Automata)

### Docker Configuration
- Python 3.10 slim base image
- System dependencies: graphviz, curl
- Non-root user for security
- Health checks every 30s
- Gunicorn with 4 workers + UvicornWorker
- Optimized layer caching for faster builds