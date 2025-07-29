# State Forge API - FastAPI ML Backend

A FastAPI backend for Automata Theory conversions using machine learning models.

## Features

- **DFA Minimization**: Convert DFA to minimized DFA
- **Regex to ε-NFA**: Convert regular expressions to epsilon-NFAs
- **ε-NFA to DFA**: Convert epsilon-NFAs to DFAs
- **Push Down Automata**: Handle PDA operations

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy environment variables:
   ```bash
   cp .env.example .env
   ```
4. Run the application:
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## Railway Deployment

### Option 1: Automatic Deployment (Recommended)

1. **Connect to Railway**:

   - Go to [Railway](https://railway.app)
   - Click "Start a New Project"
   - Connect your GitHub repository

2. **Configure Environment Variables**:
   In your Railway project dashboard, add these environment variables:

   ```
   DEBUG=false
   ENVIRONMENT=production
   ALLOWED_ORIGINS=https://yourdomain.com
   PYTHONUNBUFFERED=1
   PIP_NO_CACHE_DIR=1
   ```

3. **Deploy**:
   Railway will automatically detect the configuration and deploy your app.

### Option 2: Docker Deployment (For Build Issues)

If you encounter build timeout errors, use Docker deployment:

1. **Enable Docker in Railway**:

   - In your Railway project settings
   - Go to "Build" section
   - Select "Docker" as the build method

2. **The Dockerfile will handle the build process automatically**

### Manual CLI Deployment

1. **Install Railway CLI**:

   ```bash
   npm install -g @railway/cli
   ```

2. **Login and Initialize**:

   ```bash
   railway login
   railway init
   ```

3. **Deploy**:
   ```bash
   railway up
   ```

## Troubleshooting Build Issues

### Build Timeout Error

If you encounter build timeout errors:

1. **Use Docker deployment** (Option 2 above)
2. **Check Railway build logs** for specific error messages
3. **Ensure all files are committed** to your Git repository
4. **Try reducing workers** in `railway.toml` if memory issues occur

### Common Solutions

1. **Clear Railway cache**:

   - In Railway dashboard → Settings → General → Clear Build Cache

2. **Use Docker for complex dependencies**:

   - The project includes both `railway.toml` and `Dockerfile`
   - Railway will automatically detect and use Docker if Nixpacks fails

3. **Optimize dependencies**:
   - Uses CPU-only PyTorch for faster builds
   - Specific dependency versions for stability
   - Excluded unnecessary packages like Streamlit

## Configuration

The application uses the following configuration files:

- `railway.toml`: Railway deployment configuration
- `Dockerfile`: Docker deployment configuration
- `nixpacks.toml`: Nixpacks build optimization
- `Procfile`: Process configuration (alternative)
- `.env`: Environment variables (local development)
- `requirements.txt`: Python dependencies

## Environment Variables

| Variable          | Description                                   | Default       |
| ----------------- | --------------------------------------------- | ------------- |
| `DEBUG`           | Enable debug mode                             | `true`        |
| `ENVIRONMENT`     | Environment name                              | `development` |
| `PORT`            | Server port (Railway sets this automatically) | `8000`        |
| `HOST`            | Server host                                   | `0.0.0.0`     |
| `ALLOWED_ORIGINS` | CORS allowed origins (comma-separated)        | `*`           |

## API Endpoints

- `GET /`: Health check
- `POST /api/v1/convert`: Convert input using ML models
- `GET /api/v1/models`: Get available models

## Production Considerations

1. **CORS**: Update `ALLOWED_ORIGINS` to restrict access to your frontend domains
2. **Debug Mode**: Set `DEBUG=false` in production
3. **Monitoring**: Railway provides built-in monitoring and logs
4. **Scaling**: Railway automatically handles scaling based on traffic

## Model Files

The application includes pre-trained models in the `models/` directory:

- DFA Minimization models (~7.2MB)
- Regex to ε-NFA models
- ε-NFA to DFA models
- Push Down Automata models

These are automatically included in the Railway deployment.

## Performance Optimizations

- **CPU-only PyTorch**: Faster builds and smaller memory footprint
- **Reduced workers**: 2 Gunicorn workers to optimize for Railway's memory limits
- **Extended timeouts**: 300-second timeouts to handle ML model loading
- **Build caching**: Optimized dependency installation order
