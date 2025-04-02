import asyncio
from typing import TypedDict, Annotated, List, AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
import operator

# Définition de l'état du graphe
class GraphState(TypedDict):
    messages: Annotated[List[dict], operator.add]
    human_input: str
    ai_response: str
    narrative_evoquee: str
    selected_themes: List[Dict[str, Any]]
    agent_id: str  # Nouvel attribut pour identifier l'agent

# Configuration OpenAI avec clé API en dur pour démonstration
from openai import AsyncOpenAI

# Clé API OpenAI directement dans le code (pour démonstration uniquement)

# Initialisation du client OpenAI
try:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    print("Client OpenAI configuré avec succès")
except Exception as e:
    print(f"Erreur de configuration OpenAI : {e}")
    client = None

# Variable globale pour stocker les morceaux de streaming
streaming_chunks = []

async def get_streaming_chunks():
    """Récupérer les chunks de streaming avec identification des agents"""
    global streaming_chunks
    
    # Vérifier si des chunks existent
    if not streaming_chunks:
        return []
    
    # Récupérer et réinitialiser les chunks
    chunks_copy = streaming_chunks.copy()
    streaming_chunks.clear()
    
    return chunks_copy

def map_node_to_agent_id(node_name):
    """Mapper les noms de nœuds à des identifiants d'agents"""
    agent_mapping = {
        'salutation_node': 'salutation_agent',
        'conversation_node': 'conversation_agent', 
        'resume_conversation_node': 'resume_agent',
        'error_node': 'error_agent'
    }
    return agent_mapping.get(node_name, 'default_agent')

async def salutation_node(state: GraphState) -> Dict[str, Any]:
    """Nœud de salutation qui génère une réponse initiale"""
    if client is None:
        ai_response = "Erreur : Configuration OpenAI invalide"
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": ai_response,
                    "agent_id": "salutation_agent"
                }
            ],
            "ai_response": ai_response,
            "agent_id": "salutation_agent"
        }
    
    # Préparer le message de salutation
    human_input = state.get('human_input', 'Bonjour')
    
    # Appel au LLM pour une salutation personnalisée
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Tu es un agent de salutation amical et accueillant."},
                {"role": "user", "content": human_input}
            ]
        )
        
        ai_response = response.choices[0].message.content
        
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": ai_response,
                    "agent_id": "salutation_agent"
                }
            ],
            "ai_response": ai_response,
            "agent_id": "salutation_agent"
        }
    
    except Exception as e:
        error_response = f"Erreur de salutation : {str(e)}"
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": error_response,
                    "agent_id": "salutation_agent"
                }
            ],
            "ai_response": error_response,
            "agent_id": "salutation_agent"
        }

async def conversation_node(state: GraphState) -> Dict[str, Any]:
    """Nœud de conversation qui génère une réponse à l'entrée utilisateur"""
    if client is None:
        ai_response = "Erreur : Configuration OpenAI invalide"
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": ai_response,
                    "agent_id": "conversation_agent"
                }
            ],
            "ai_response": ai_response,
            "agent_id": "conversation_agent"
        }
    
    # Récupérer l'historique des messages
    messages = state.get('messages', [])
    
    # Sélectionner uniquement le dernier message utilisateur
    last_user_message = next((msg for msg in reversed(messages) if msg['role'] == 'user'), None)
    
    # Préparer les messages pour le contexte
    context_messages = [
        {"role": "system", "content": "Tu es un assistant conversationnel français, attentif et précis."}
    ]
    
    # Ajouter uniquement le dernier message utilisateur
    if last_user_message:
        context_messages.append(last_user_message)
    
    try:
        # Appel au LLM pour générer une réponse
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=context_messages
        )
        
        # Récupérer la réponse complète
        ai_response = response.choices[0].message.content
        
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": ai_response,
                    "agent_id": "conversation_agent"
                }
            ],
            "ai_response": ai_response,
            "agent_id": "conversation_agent"
        }
    
    except Exception as e:
        error_response = f"Erreur de conversation : {str(e)}"
        return {
            "messages": [
                {
                    "role": "assistant", 
                    "content": error_response,
                    "agent_id": "conversation_agent"
                }
            ],
            "ai_response": error_response,
            "agent_id": "conversation_agent"
        }

async def resume_conversation_node(state: GraphState) -> Dict[str, Any]:
    """Nœud qui résume la conversation précédente"""
    if client is None:
        ai_response = "Erreur : Configuration OpenAI invalide"
        return {
            "messages": [{"role": "assistant", "content": ai_response}],
            "ai_response": ai_response
        }
    
    # Récupérer l'historique des messages
    messages = state.get('messages', [])
    
    try:
        # Préparer les messages pour le résumé
        context_messages = [
            {"role": "system", "content": "Résume brièvement la conversation précédente en français."}
        ] + messages
        
        # Appel au LLM pour résumer
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
                {"role": "assistant", "content": f"Résumé de la conversation : {ai_response}"}
            ],
            "ai_response": ai_response
        }
    except Exception as e:
        error_msg = f"Erreur de résumé : {str(e)}"
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
    workflow.add_node("resume_conversation", resume_conversation_node)
    
    # Configuration des transitions
    workflow.set_entry_point("salutation")
    workflow.add_edge("salutation", "conversation")
    workflow.add_edge("conversation", "resume_conversation")
    workflow.add_edge("resume_conversation", END)
    
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
        yield {
            'content': "Erreur : Configuration OpenAI invalide",
            'agent_id': 'error_agent'
        }
        return
    
    # Créer et exécuter le graphe
    graph = create_graph()
    
    # Initialiser l'état avec l'entrée utilisateur
    initial_state = {"human_input": human_input}
    
    try:
        # Exécution du graphe
        async for state in graph.astream(initial_state):
            # Récupérer le nom du nœud actuel
            current_node = graph.get_state().get('__running_node')
            agent_id = map_node_to_agent_id(current_node)
            
            # Récupérer les morceaux de streaming
            chunks = await get_streaming_chunks()
            for chunk in chunks:
                yield {
                    'content': chunk,
                    'agent_id': agent_id
                }
                await asyncio.sleep(0.01)  # Pause légère pour un meilleur effet visuel
    
    except Exception as e:
        error_message = f"Erreur lors de l'exécution du graphe : {str(e)}"
        print(error_message)
        yield {
            'content': error_message,
            'agent_id': 'error_agent'
        }
