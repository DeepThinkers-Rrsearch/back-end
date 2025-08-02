import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.models.schemas import ConversionRequest, ConversionResponse, HealthResponse, ModelType
from app.services.conversion_service import conversion_service

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

app = FastAPI(
    title=settings.app_name,
    description="ML Model Inference API for Automata Theory Conversions",
    version="1.0.0",
    debug=settings.debug
)

logger.info(f"Starting {settings.app_name} in {settings.environment} mode")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for Railway deployment"""
    try:
        logger.info("Health check endpoint called")
        
        # Basic health check
        health_data = HealthResponse(
            status="healthy",
            available_models=[model.value for model in ModelType]
        )
        
        # Log successful health check
        logger.info("Health check passed successfully")
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Service unavailable: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check_alt():
    """Alternative health check endpoint"""
    return await health_check()

@app.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity test"""
    return {"status": "ok", "message": "pong"}

@app.post("/api/v1/convert", response_model=ConversionResponse)
async def convert_input(request: ConversionRequest):
    """
    Convert input using specified ML model and return result with diagram
    
    - **input_text**: The text to convert (regex, DFA description, etc.)
    - **model_type**: The conversion model to use
    """
    
    try:
        # Validate input
        if not conversion_service.validate_input(request.input_text, request.model_type):
            return ConversionResponse(
                success=False,
                error="Invalid input: Text cannot be empty or too long"
            )
        
        logger.info(f"Converting with {request.model_type.value}: {request.input_text[:50]}...")
        
        # Perform conversion
        result, isAccepted = conversion_service.convert(
            request.input_text, 
            request.model_type
        )
        
        logger.info(f"Conversion successful for {request.model_type.value}")
        
        return ConversionResponse(
            success=True,
            result=result,
            isAccepted=isAccepted
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return ConversionResponse(
            success=False,
            error=f"Input validation failed: {str(e)}"
        )
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        return ConversionResponse(
            success=False,
            error=f"Model not available: {str(e)}"
        )
        
    except Exception as e:
        logger.error(f"Conversion error: {e}")
        return ConversionResponse(
            success=False,
            error=f"Conversion failed: {str(e)}"
        )

@app.get("/api/v1/models")
async def get_available_models():
    """Get list of available models"""
    return {
        "models": [
            {
                "name": model.value,
                "description": f"Convert using {model.value} model"
            }
            for model in ModelType
        ]
    }

