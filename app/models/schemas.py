from pydantic import BaseModel
from enum import Enum
from typing import Optional

class ModelType(str, Enum):
    DFA_MINIMIZATION = "DFA-Minimization"
    REGEX_TO_E_NFA = "Regex-to-ε-NFA"
    E_NFA_TO_DFA = "e_NFA-to-DFA"
    PDA = "PDA"

class ConversionRequest(BaseModel):
    input_text: str
    model_type: ModelType
    
class ConversionResponse(BaseModel):
    success: bool
    result: Optional[str] = None
    isAccepted: bool
    # diagram_base64: Optional[str] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    available_models: list[str]
