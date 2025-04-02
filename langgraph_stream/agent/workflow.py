"""
Workflow LangGraph avec streaming
"""

import asyncio
from typing import AsyncGenerator

# Fonction simple pour dire bonjour avec streaming
async def dire_bonjour_stream() -> AsyncGenerator[str, None]:
    """Fonction qui dit bonjour en streaming"""
    message = "Bonjour"
    for char in message:
        yield char
        await asyncio.sleep(0.05)

# Fonction simple pour dire bonsoir avec streaming
async def dire_bonsoir_stream() -> AsyncGenerator[str, None]:
    """Fonction qui dit bonsoir en streaming"""
    message = "Bonsoir"
    for char in message:
        yield char
        await asyncio.sleep(0.05)

# Fonction pour exécuter le workflow en mode streaming
async def execute_graph_stream(message_type: str = "all") -> AsyncGenerator[str, None]:
    """Exécute le workflow et retourne un générateur pour le streaming
    
    Args:
        message_type: Type de message à générer ("bonjour", "bonsoir" ou "all")
    """
    
    if message_type in ["bonjour", "all"]:
        # Streaming de bonjour
        async for char in dire_bonjour_stream():
            yield char
    
    if message_type == "all":
        # Petit délai entre les deux messages si on affiche les deux
        await asyncio.sleep(0.5)
    
    if message_type in ["bonsoir", "all"]:
        # Streaming de bonsoir
        async for char in dire_bonsoir_stream():
            yield char
