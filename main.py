from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.models.schemas import ConversionRequest, ConversionResponse, HealthResponse, ModelType
from app.services.conversion_service import conversion_service
from app.core.config import settings
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="ML Model Inference API for Automata Theory Conversions",
    version="1.0.0",
    debug=settings.debug
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        return HealthResponse(
            status="healthy",
            available_models=[model.value for model in ModelType]
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")

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
        result, diagram_base64 = conversion_service.convert(
            request.input_text, 
            request.model_type
        )
        
        logger.info(f"Conversion successful for {request.model_type.value}")
        
        return ConversionResponse(
            success=True,
            result=result,
            diagram_base64=diagram_base64
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

if __name__ == "__main__":
    import uvicorn
    # Use Railway's PORT environment variable, fallback to 8000 for local development
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port, 
        reload=settings.debug,
        log_level="info"
    )