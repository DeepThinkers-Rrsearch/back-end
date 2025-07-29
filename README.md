# State Forge API - Railway Deployment

## Overview
FastAPI ML backend for Automata Theory Conversions with 4 PyTorch models for DFA minimization, regex-to-ε-NFA, ε-NFA-to-DFA, and PDA conversions.

## Railway Deployment

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
- `Procfile` - Gunicorn server configuration
- `railway.toml` - Railway deployment settings
- `nixpacks.toml` - Python build configuration
- `.railwayignore` - Files to exclude from deployment

### API Endpoints
- `GET /` - Health check
- `POST /api/v1/convert` - Model conversion endpoint
- `GET /api/v1/models` - List available models

### Models Included
- DFA Minimization (~7.2MB)
- Regex-to-ε-NFA
- ε-NFA-to-DFA  
- PDA (Push Down Automata)

### Production Settings
- Uses gunicorn with 4 workers
- UvicornWorker for async support
- Health check timeout: 300s
- Restart policy: ON_FAILURE with 10 max retries