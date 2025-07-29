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

### Automatic Deployment

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
   ```

3. **Deploy**:
   Railway will automatically detect the configuration and deploy your app.

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

## Configuration

The application uses the following configuration files:

- `railway.toml`: Railway deployment configuration
- `Procfile`: Process configuration (alternative to railway.toml)
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
