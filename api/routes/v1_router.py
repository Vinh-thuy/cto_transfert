from fastapi import APIRouter
import logging

# Importer les routes existantes
from .agent_langgraph_router import router as langgraph_router

# Configuration du logger
logger = logging.getLogger(__name__)

# Créer le routeur principal V1
v1_router = APIRouter(prefix="/v1")

# Inclure les routes
v1_router.include_router(langgraph_router, prefix="/agents")

logger.info("Routes V1 initialisées avec succès")