from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import httpx
import os
from dotenv import load_dotenv
from agno.embedder.base import Embedder
from agno.utils.log import logger

# Charger les variables d'environnement
load_dotenv()

@dataclass
class InfinityEmbedder(Embedder):
    id: str = "default-model-id"  # Remplace par l'ID du modèle Infinity
    dimensions: int = 1536  # Ajuste en fonction du modèle utilisé
    encoding_format: str = "float"  # Ou "base64"
    api_url: str = os.getenv("INFINITY_API_URL", "http://localhost:8000")  # URL de ton endpoint Infinity
    api_key: str = os.getenv("INFINITY_API_KEY")  # Clé API depuis les variables d'environnement
    request_params: Optional[Dict[str, Any]] = None
    client: Optional[httpx.Client] = None

    def __post_init__(self):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        self.client = httpx.Client(base_url=self.api_url, headers=headers, verify=False)

    def response(self, text: str) -> Dict[str, Any]:
        _request_params: Dict[str, Any] = {
            "model": self.id,
            "input": text,
            "encoding_format": self.encoding_format,
        }
        if self.request_params:
            _request_params.update(self.request_params)
        try:
            response = self.client.post("/v1/embeddings", json=_request_params)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            logger.warning(f"Request to Infinity API failed: {e}")
            return {}

    def get_embedding(self, text: str) -> List[float]:
        response_data = self.response(text=text)
        try:
            return response_data['data'][0]['embedding']
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to retrieve embedding: {e}")
            return []

    def get_embedding_and_usage(self, text: str) -> Tuple[List[float], Optional[Dict]]:
        response_data = self.response(text=text)
        try:
            embedding = response_data['data'][0]['embedding']
            usage = response_data.get('usage')
            return embedding, usage
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to retrieve embedding and usage: {e}")
            return [], None