from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    app_name: str = "State Forge API"
    debug: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "yes")
    api_v1_prefix: str = "/api/v1"
    environment: str = os.getenv("ENVIRONMENT", "development")

    # Model paths
    models_root: str = "./models"

    dfa_minimization_model_path: str = "models/dfa_minimization/dfa_minimizer_10_new_transformer.pt"
    dfa_minimization_blob_model_path: str = "dfa_minimization/dfa_minimizer_10_new_transformer.pt"
    dfa_minimization_tokenizer_path: str = "models/dfa_minimization/dfa_minimizer_10_new_tokenizer.pkl"
    dfa_minimization_blob_tokenizer_path: str = "dfa_minimization/dfa_minimizer_10_new_tokenizer.pkl"

    regex_to_e_nfa_model_path: str = "models/regex_to_e_nfa/transformer_regex_to_e_nfa.pt"
    regex_to_e_nfa_blob_model_path: str = "regex_to_e_nfa/transformer_regex_to_e_nfa.pt"
    regex_to_e_nfa_tokenizer_path: str = "models/regex_to_e_nfa/regex_to_e_nfa_tokenizer.pkl"
    regex_to_e_nfa_blob_tokenizer_path: str = "regex_to_e_nfa/regex_to_e_nfa_tokenizer.pkl"
    

    e_nfa_to_dfa_model_path: str = "models/e_nfa_to_dfa/transformer_model.pt"
    e_nfa_to_dfa_blob_model_path: str = "e_nfa_to_dfa/transformer_model.pt"
    e_nfa_to_dfa_tokenizer_path: str = "models/e_nfa_to_dfa/e_nfa_to_dfa_tokenizer.pkl"
    e_nfa_to_dfa_blob_tokenizer_path: str = "e_nfa_to_dfa/e_nfa_to_dfa_tokenizer.pkl"
    
    pda_model_path: str = "models/pda/pda.pth"
    pda_blob_model_path: str = "pda/pda.pth"
    pda_tokenizer_path: str = "models/pda/pda_tokenizer.pkl"
    pda_blob_tokenizer_path: str = "pda/pda_tokenizer.pkl"

    # CORS settings
    allowed_origins: List[str] = [
        origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",")
    ]

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
