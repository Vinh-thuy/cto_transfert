"""
Nœuds pour le workflow LangGraph
"""

import asyncio
from typing import Dict, Any, AsyncIterator

# Fonctions de nœuds pour le streaming
async def dire_bonjour(state: Dict[str, Any]) -> AsyncIterator:
    """Nœud qui dit bonjour"""
    message = "Bonjour"
    
    # Streaming du message caractère par caractère
    for char in message:
        # Simulation d'un délai pour mieux visualiser le streaming
        await asyncio.sleep(0.05)
        # Yield pour le streaming
        yield {"node_output": char}
    
    # Création d'une copie de l'état pour éviter les modifications directes
    new_state = state.copy()
    
    # Mise à jour des messages dans l'état
    if "messages" not in new_state:
        new_state["messages"] = []
    else:
        # Création d'une copie de la liste pour éviter les modifications directes
        new_state["messages"] = new_state["messages"].copy()
    
    new_state["messages"].append(message)
    yield new_state

async def dire_bonsoir(state: Dict[str, Any]) -> AsyncIterator:
    """Nœud qui dit bonsoir"""
    message = "Bonsoir"
    
    # Streaming du message caractère par caractère
    for char in message:
        # Simulation d'un délai pour mieux visualiser le streaming
        await asyncio.sleep(0.05)
        # Yield pour le streaming
        yield {"node_output": char}
    
    # Création d'une copie de l'état pour éviter les modifications directes
    new_state = state.copy()
    
    # Mise à jour des messages dans l'état
    if "messages" not in new_state:
        new_state["messages"] = []
    else:
        # Création d'une copie de la liste pour éviter les modifications directes
        new_state["messages"] = new_state["messages"].copy()
    
    new_state["messages"].append(message)
    yield new_state
