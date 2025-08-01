import base64
from typing import Tuple
import logging
from app.models.schemas import ModelType
from app.services.model_service import model_service
from utils.push_down_automata import simulate_pda

logger = logging.getLogger(__name__)

class ConversionService:
    
    def __init__(self):
        logger.info("ConversionService initialized (prediction functions will load on demand)")
    
    def _lazy_import_prediction_functions(self):
        """Import prediction functions only when needed to avoid startup delays"""
        try:
            global predict_dfa_minimization, predict_regex_to_e_nfa, predict_e_nfa_to_dfa, predict_PDA_transitions
            global epsilon_nfa_to_dot, minimized_dfa_to_dot, dfa_output_to_dot, pda_output_to_dot
            
            from utils.dfa_minimization import predict_dfa_minimization
            from utils.regex_to_epsilon_nfa import predict_regex_to_e_nfa
            from utils.e_nfa_to_dfa import predict_e_nfa_to_dfa
            from utils.push_down_automata import predict_PDA_transitions
            from utils.graphviz.graphviz_regex_to_e_nfa import epsilon_nfa_to_dot
            from utils.graphviz.graphviz_minimized_dfa import minimized_dfa_to_dot
            from utils.graphviz.graphviz_dfa import dfa_output_to_dot
            from utils.graphviz.graphviz_pda import pda_output_to_dot
            
            logger.info("Prediction and graphviz functions imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import prediction functions: {e}")
            raise
    
    def validate_input(self, input_text: str, model_type: ModelType) -> bool:
        """Validate input without requiring heavy imports"""
        if not input_text or not input_text.strip():
            return False
        if len(input_text.strip()) > 10000:  # Reasonable limit
            return False
        return True
    
    def convert(self, input_text: str, model_type: ModelType) -> Tuple[str, str]:
        """
        Convert input using specified model and return result + base64 diagram
        
        Args:
            input_text: The input string to convert
            model_type: The type of model to use for conversion
            
        Returns:
            Tuple of (conversion_result, base64_encoded_diagram)
        """
        
        if not input_text or not input_text.strip():
            raise ValueError("Input text cannot be empty")
            
        # Import prediction functions lazily
        self._lazy_import_prediction_functions()
        
        logger.info(f"Starting conversion for {model_type.value}")
            
        try:
            # Load the appropriate model
            model, stoi, itos = model_service.load_model(model_type.value)
            
            # Perform conversion based on model type
            if model_type == ModelType.REGEX_TO_E_NFA:
                result = predict_regex_to_e_nfa(input_text, model, stoi, itos)
                #graph = epsilon_nfa_to_dot(result)
                isAccepted = True
                
            elif model_type == ModelType.DFA_MINIMIZATION:
                result = predict_dfa_minimization(model, input_text)
                #graph = minimized_dfa_to_dot(result)
                isAccepted = True
                
            elif model_type == ModelType.E_NFA_TO_DFA:
                result = predict_e_nfa_to_dfa(model, input_text)
                #graph = dfa_output_to_dot(result)
                isAccepted = True
                
            elif model_type == ModelType.PDA:
                transitions_list = predict_PDA_transitions(model, input_text)
                # Convert list of transitions to single string for response
                result = '\n'.join(transitions_list) if transitions_list else 'No valid transitions found'
                # Use original list for graph generation
                #graph = pda_output_to_dot(transitions_list)
                isAccepted = simulate_pda(input_text,transitions_list)
                
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Generate diagram and convert to base64
            # try:
            #     if(model_type!="PDA"):
            #         png_bytes = graph.pipe(format="png")
            #         diagram_base64 = base64.b64encode(png_bytes).decode('utf-8')
            #     else:
            #         diagram_base64 = graph
            # except Exception as e:
            #     raise Exception(f"Failed to generate diagram: {str(e)}")
            
            return result, isAccepted
            
        except Exception as e:
            logger.error(f"Conversion failed for {model_type.value}: {str(e)}")
            raise Exception(f"Conversion failed for {model_type.value}: {str(e)}")

# Create global instance
conversion_service = ConversionService()
