import asyncio
from typing import TypedDict, Annotated, List, AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
import operator

# Configuration OpenAI avec clé API en dur pour démonstration
from openai import AsyncOpenAI

# Clé API OpenAI directement dans le code (pour démonstration uniquement)


# Initialisation du client OpenAI
try:
    client = AsyncOpenAI(api_key=API_KEY)
    print("Client OpenAI configuré avec succès")
except Exception as e:
    print(f"Erreur de configuration OpenAI : {e}")
    client = None

# Définition de l'état du graphe
class GraphState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    human_input: str
    ai_response: str

# Variable globale pour stocker les morceaux de streaming
streaming_chunks = []

def get_streaming_chunks():
    """Récupère les morceaux de streaming et vide la liste"""
    global streaming_chunks
    chunks = streaming_chunks.copy()
    streaming_chunks = []
    return chunks

async def salutation_node(state: GraphState) -> Dict[str, Any]:
    """Nœud de salutation qui génère une réponse initiale"""
    if client is None:
        ai_response = "Erreur : Configuration OpenAI invalide"
        return {
            "messages": [{"role": "assistant", "content": ai_response}],
            "ai_response": ai_response
        }
    
    # Préparer le message de salutation
    human_input = state.get('human_input', 'Bonjour')
    
    # Appel au LLM pour une salutation personnalisée
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un assistant français chaleureux."},
                {"role": "user", "content": f"Salutation en réponse à : {human_input}"}
            ],
            stream=True
        )
        
        # Extraire la réponse en mode streaming
        ai_response = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                ai_response += content
                # Stocker le morceau pour le streaming
                global streaming_chunks
                streaming_chunks.append(content)
        
        # Retourner l'état final
        return {
            "messages": [
                {"role": "user", "content": human_input},
                {"role": "assistant", "content": ai_response}
            ],
            "ai_response": ai_response
        }
    except Exception as e:
        error_msg = f"Erreur LLM : {str(e)}"
        return {
            "messages": [{"role": "assistant", "content": error_msg}],
            "ai_response": error_msg
        }

async def conversation_node(state: GraphState) -> Dict[str, Any]:
    """Nœud de conversation qui approfondit l'échange"""
    if client is None:
        ai_response = "Erreur : Configuration OpenAI invalide"
        return {
            "messages": [{"role": "assistant", "content": ai_response}],
            "ai_response": ai_response
        }
    
    # Récupérer l'historique des messages
    messages = state.get('messages', [])
    
    # Préparer les messages pour le contexte
    try:
        context_messages = [
            {"role": "system", "content": "Tu es un assistant français conversationnel."}
        ] + messages
        
        # Appel au LLM pour continuer la conversation
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context_messages,
            stream=True
        )
        
        # Extraire la réponse en mode streaming
        ai_response = ""
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                ai_response += content
                # Stocker le morceau pour le streaming
                global streaming_chunks
                streaming_chunks.append(content)
        
        # Retourner l'état final
        return {
            "messages": [
                {"role": "assistant", "content": ai_response}
            ],
            "ai_response": ai_response
        }
    except Exception as e:
        error_msg = f"Erreur LLM : {str(e)}"
        return {
            "messages": [{"role": "assistant", "content": error_msg}],
            "ai_response": error_msg
        }

def create_graph():
    """Crée et configure le workflow LangGraph"""
    workflow = StateGraph(GraphState)
    
    # Ajout des nœuds
    workflow.add_node("salutation", salutation_node)
    workflow.add_node("conversation", conversation_node)
    
    # Configuration des transitions
    workflow.set_entry_point("salutation")
    workflow.add_edge("salutation", "conversation")
    workflow.add_edge("conversation", END)
    
    # Compilation du graphe
    return workflow.compile()

# Fonction pour exécuter le workflow en mode streaming
async def execute_graph_stream(human_input: str = "Bonjour") -> AsyncGenerator[str, None]:
    """Exécute le workflow et retourne un générateur pour le streaming
    
    Args:
        human_input: Message initial de l'utilisateur
    """
    # Vérification de la configuration
    if client is None:
        yield "Erreur : Configuration OpenAI invalide"
        return
    
    # Créer et exécuter le graphe
    graph = create_graph()
    
    # Initialiser l'état avec l'entrée utilisateur
    initial_state = {"human_input": human_input}
    
    try:
        # Exécution du graphe
        async for state in graph.astream(initial_state):
            # Récupérer les morceaux de streaming
            chunks = get_streaming_chunks()
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)  # Pause légère pour un meilleur effet visuel
    except Exception as e:
        error_message = f"Erreur lors de l'exécution du graphe : {str(e)}"
        print(error_message)
        yield error_message
