from typing import Dict, Any, Tuple, Optional
import os
import logging
from app.core.config import settings

# Import model loading functions lazily to avoid startup delays
logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
        self._loaded_models = set()
        logger.info("ModelService initialized (models will load on demand)")
        
    def _validate_model_path(self, path: str) -> bool:
        """Validate if model file exists"""
        exists = os.path.exists(path)
        if not exists:
            logger.warning(f"Model file not found: {path}")
        return exists
        
    def _lazy_import_model_functions(self):
        """Import model loading functions only when needed"""
        try:
            global load_dfa_minimization_model, load_regex_to_e_nfa_model, load_e_nfa_to_dfa_model, load_PDA_model
            from utils.dfa_minimization import load_dfa_minimization_model
            from utils.regex_to_epsilon_nfa import load_regex_to_e_nfa_model
            from utils.e_nfa_to_dfa import load_e_nfa_to_dfa_model
            from utils.push_down_automata import load_PDA_model
            logger.info("Model loading functions imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import model loading functions: {e}")
            raise
        
    def load_model(self, model_type: str) -> Tuple[Any, Optional[Any], Optional[Any]]:
        """Load and cache models for inference"""
        
        # Import model functions lazily
        self._lazy_import_model_functions()
        
        # Return cached model if already loaded
        cache_key = model_type
        if cache_key in self._models:
            stoi = self._tokenizers.get(f"{model_type}_stoi")
            itos = self._tokenizers.get(f"{model_type}_itos")
            logger.info(f"Using cached model: {model_type}")
            return self._models[cache_key], stoi, itos
            
        logger.info(f"Loading model: {model_type}")
        
        try:
            if model_type == "DFA-Minimization":
                if not self._validate_model_path(settings.dfa_minimization_model_path):
                    raise FileNotFoundError(f"DFA model not found: {settings.dfa_minimization_model_path}")
                    
                model = load_dfa_minimization_model(
                    settings.dfa_minimization_model_path,
                    settings.dfa_minimization_tokenizer_path
                )
                self._models[cache_key] = model
                logger.info(f"Successfully loaded DFA minimization model")
                return model, None, None
                
            elif model_type == "Regex-to-ε-NFA":
                if not self._validate_model_path(settings.regex_to_e_nfa_model_path):
                    raise FileNotFoundError(f"Regex model not found: {settings.regex_to_e_nfa_model_path}")
                    
                model, stoi, itos = load_regex_to_e_nfa_model(
                    settings.regex_to_e_nfa_model_path,
                    settings.regex_to_e_nfa_tokenizer_path
                )
                self._models[cache_key] = model
                self._tokenizers[f"{model_type}_stoi"] = stoi
                self._tokenizers[f"{model_type}_itos"] = itos
                logger.info(f"Successfully loaded Regex-to-ε-NFA model")
                return model, stoi, itos
                
            elif model_type == "e_NFA-to-DFA":
                if not self._validate_model_path(settings.e_nfa_to_dfa_model_path):
                    raise FileNotFoundError(f"e-NFA model not found: {settings.e_nfa_to_dfa_model_path}")
                    
                model = load_e_nfa_to_dfa_model(settings.e_nfa_to_dfa_model_path)
                self._models[cache_key] = model
                logger.info(f"Successfully loaded e-NFA-to-DFA model")
                return model, None, None
                
            elif model_type == "PDA":
                if not self._validate_model_path(settings.pda_model_path):
                    raise FileNotFoundError(f"PDA model not found: {settings.pda_model_path}")
                    
                model = load_PDA_model(settings.pda_model_path)
                self._models[cache_key] = model
                logger.info(f"Successfully loaded PDA model")
                return model, None, None
                
            else:
                raise ValueError(f"Unknown model type: {model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_type}: {str(e)}")
            raise Exception(f"Failed to load model {model_type}: {str(e)}")
    
    def get_loaded_models(self) -> list:
        """Get list of currently loaded models"""
        return list(self._models.keys())
    
    def clear_cache(self):
        """Clear model cache"""
        self._models.clear()
        self._tokenizers.clear()
        logger.info("Model cache cleared")

# Create global instance
model_service = ModelService()
