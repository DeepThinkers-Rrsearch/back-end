from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    app_name: str = "State Forge API"
    debug: bool = os.getenv("DEBUG", "true").lower() == "true"
    api_v1_prefix: str = "/api/v1"
    environment: str = os.getenv("ENVIRONMENT", "development")
    
    # Model paths
    models_root: str = "./models"
    dfa_minimization_model_path: str = "models/dfa_minimization/dfa_minimizer_transformer.pt"
    dfa_minimization_tokenizer_path: str = "models/dfa_minimization/dfa_minimizer_tokenizer.pkl"
    regex_to_e_nfa_model_path: str = "models/regex_to_e_nfa/transformer_regex_to_e_nfa.pt"
    regex_to_e_nfa_tokenizer_path: str = "models/regex_to_e_nfa/regex_to_e_nfa_tokenizer.pkl"
    e_nfa_to_dfa_model_path: str = "models/e_nfa_to_dfa/transformer_model.pt"
    pda_model_path: str = "models/pda/pda.pth"
    
    # CORS settings - more restrictive for production
    allowed_origins: List[str] = (
        os.getenv("ALLOWED_ORIGINS", "*").split(",") 
        if os.getenv("ALLOWED_ORIGINS") 
        else ["*"]
    )
    
    # Railway specific settings
    port: int = int(os.getenv("PORT", 8000))
    host: str = os.getenv("HOST", "0.0.0.0")
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
