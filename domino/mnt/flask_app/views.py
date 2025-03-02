from flask_app import app, api
from flask_restx import Resource, fields
from flask import request
from vllm import LLM, SamplingParams
import torch
import platform
import logging
import os
import warnings

# Désactiver certains warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration pour contourner les problèmes OpenMP
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

# Namespace pour l'API
ns = api.namespace('chat', description='Opérations de chat')

# Modèle de requête
chat_model = api.model('ChatRequest', {
    'messages': fields.List(fields.Raw, required=True, description='Liste des messages'),
    'max_tokens': fields.Integer(default=500, description='Nombre maximum de tokens'),
    'temperature': fields.Float(default=0.7, description='Température de sampling')
})

# Modèle de réponse
response_model = api.model('ChatResponse', {
    'response': fields.String(description='Réponse générée')
})

def load_model_safely():
    """
    Chargement sécurisé du modèle avec gestion des erreurs
    """
    try:
        # Configuration matérielle
        is_apple_silicon = platform.machine() == 'arm64'
        cuda_available = torch.cuda.is_available()
        
        logger.info(f"Configuration détectée : Apple Silicon = {is_apple_silicon}, CUDA = {cuda_available}")
        
        # Paramètres adaptés à l'architecture
        model_params = {
            'model': "microsoft/Phi-4-mini-instruct", 
            'trust_remote_code': True, 
            'max_model_len': 32768,
            'download_dir': "/Users/vinh/Documents/litellm/models",
            'enforce_eager': True,  # Force mode eager pour éviter les problèmes de compilation
            'tensor_parallel_size': 1,  # Limiter le parallélisme
            'gpu_memory_utilization': 0.5  # Utilisation mémoire réduite
        }
        
        # Ajustements spécifiques pour Mac ARM
        if is_apple_silicon:
            model_params.update({
                'dtype': torch.float16,
                'device': 'cpu'
            })
        elif cuda_available:
            model_params.update({
                'dtype': torch.float16,
                'device': 'cuda'
            })
        
        logger.info("Chargement du modèle avec paramètres : %s", model_params)
        
        return LLM(**model_params)
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise

# Chargement global du modèle
try:
    global_llm = load_model_safely()
except Exception as e:
    global_llm = None
    logger.error("Impossible de charger le modèle globalement")

@app.route('/')
def home():
    return jsonify({
        "message": "API Phi-4 Mini pour Domino DataLab",
        "endpoints": ["/chat"]
    })

@ns.route('/')
class ChatEndpoint(Resource):
    @ns.expect(chat_model)
    @ns.marshal_with(response_model)
    def post(self):
        """Générer une réponse à partir des messages"""
        if global_llm is None:
            return {"error": "Le modèle n'a pas pu être chargé"}, 500
        
        data = request.json
        messages = data.get('messages', [])
        
        # Paramètres de génération
        sampling_params = SamplingParams(
            max_tokens=data.get('max_tokens', 500),
            temperature=data.get('temperature', 0.7)
        )
        
        # Générer la réponse
        try:
            output = global_llm.chat(
                messages=messages, 
                sampling_params=sampling_params
            )
            
            return {
                'response': output[0].outputs[0].text
            }
        except Exception as e:
            logger.error(f"Erreur lors de la génération : {e}")
            return {"error": str(e)}, 500
