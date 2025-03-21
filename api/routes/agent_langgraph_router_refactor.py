from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import uuid
import logging
import traceback
from pydantic import BaseModel
from agents.LangGraph_agent import router_workflow, joke_workflow, welcome  # Import absolu depuis la racine

# Stockage des états de conversation pour chaque utilisateur
user_states = {}

# Configuration des logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter(tags=["LangGraph Agent"])

# Modèle Pydantic pour simplifier l'entrée dans Swagger
class LangGraphInput(BaseModel):
    input_data: str = "Bonjour"

@router.post("/langgraph")
async def run_langgraph_agent(input_data: LangGraphInput):
    """Endpoint POST pour exécuter le workflow LangGraph avec une entrée simplifiée"""
    # Convertir en format attendu par le workflow
    workflow_input = {"input": input_data.input_data}
    state = router_workflow.invoke(workflow_input)
    return {"result": state["output"]}

# Fonctions auxiliaires pour la gestion des requêtes WebSocket
def determine_request_type(user_id, input_data):
    """Détermine le type de requête en fonction de l'entrée et de l'état de l'utilisateur."""
    # Si l'utilisateur est en mode blague
    if user_id in user_states and user_states[user_id].get('in_joke_mode', False):
        # Si l'utilisateur veut quitter le mode blague
        if is_exit_command(input_data.get('input_data', '')):
            return "EXIT_JOKE_MODE"
        else:
            return "JOKE_MODE"
    
    # Cas par défaut - requête standard
    return "STANDARD_REQUEST"

def is_exit_command(text):
    """Vérifie si le texte est une commande de sortie."""
    if not text:
        return False
    return text.lower().strip() in ['terminette', 'stop', 'fin', 'quitter', 'terminer']

async def handle_welcome(user_id, websocket):
    """Envoie un message d'accueil à un nouvel utilisateur."""
    logging.info(f"Envoi du message d'accueil pour le nouvel utilisateur {user_id}")
    welcome_state = welcome({"input": "", "is_first_interaction": True})
    
    if "output" in welcome_state:
        welcome_message = welcome_state["output"]
        welcome_response = {
            "result": welcome_message,
            "agent_name": "Assistant Virtuel",
            "is_welcome": True
        }
        logging.info(f"Message d'accueil généré: {welcome_message}")
        await websocket.send_text(json.dumps(welcome_response))
        logging.info("Message d'accueil envoyé avec succès")

async def handle_joke_mode(user_id, input_data, websocket, config):
    """Gère les requêtes en mode blague."""
    original_input = input_data.get('input_data', '')
    logging.info(f"Utilisateur {user_id} continue en mode blague")
    joke_prompt = f"Raconte-moi une blague sur le thème: {original_input}. Fais en sorte qu'elle soit drôle et adaptée à tous les publics."
    config["recursion_limit"] = 10
    
    try:
        state = joke_workflow.invoke({"input": joke_prompt}, config=config)
        logging.info(f"Blague générée avec succès sur le thème '{original_input}'")
    except Exception as e:
        logging.error(f"Erreur lors de la génération de la blague: {str(e)}")
        state = {"output": f"Désolé, je n'ai pas pu générer une blague sur le thème '{original_input}'. Pouvez-vous essayer un autre thème? (ou tapez 'terminette' pour quitter le mode blague)"}
    
    # Ajouter un message pour indiquer à l'utilisateur qu'il peut continuer à demander des blagues
    joke_output = state.get("output", "Désolé, je n'ai pas pu générer de blague.")
    continuation_message = "\n\nVous pouvez me demander une autre blague sur un thème différent, ou taper 'terminette' pour quitter le <span style='color: green; font-weight: bold;'>mode</span> blague."
    complete_output = joke_output + continuation_message
    
    response = {
        "result": complete_output,
        "agent_name": "Humoriste Virtuel"
    }
    
    await websocket.send_text(json.dumps(response))

async def handle_exit_joke_mode(user_id, websocket):
    """Gère la sortie du mode blague."""
    logging.info(f"Utilisateur {user_id} a demandé de terminer le mode blague")
    user_states[user_id]['in_joke_mode'] = False
    response = {
        "result": "D'accord, j'arrête les blagues. Que puis-je faire d'autre pour vous ?",
        "agent_name": "Assistant Virtuel"
    }
    await websocket.send_text(json.dumps(response))

async def handle_standard_request(user_id, input_data, websocket, config, is_first_interaction=False):
    """Gère les requêtes standard en invoquant le workflow principal."""
    workflow_input = {'input': input_data.get('input_data', '')}
    
    # Si c'est la première interaction, ajouter le flag is_first_interaction
    if is_first_interaction:
        workflow_input["is_first_interaction"] = True
        logging.info(f"Première interaction détectée pour {user_id}, ajout du flag is_first_interaction")
    
    logging.info(f"Appel de router_workflow.invoke avec {workflow_input} (config: {config})")
    state = router_workflow.invoke(workflow_input, config=config)
    logging.info(f"Résultat obtenu: {state}")
    
    if "output" not in state:
        logging.error(f"Clé 'output' manquante dans l'état: {state}")
        response = {"error": "Erreur interne: format de réponse invalide", "details": str(state)}
    else:
        # Vérifier si nous devons passer en mode blague
        if "decision" in state and state["decision"] == "joke":
            # Activer le mode blague
            user_states[user_id]['in_joke_mode'] = True
            user_states[user_id]['waiting_for_theme'] = True
            
            # Ajouter un message pour indiquer à l'utilisateur qu'il est en mode blague
            state["output"] += "\n\nVous êtes maintenant en <span style='color: green; font-weight: bold;'>mode blague</span>. Donnez-moi un thème pour une blague!"
        
        response = {
            "result": state["output"],
            "agent_name": state.get("agent_name", "Assistant Virtuel")
        }
    
    await websocket.send_text(json.dumps(response))

@router.websocket("/langgraph/ws")
async def websocket_langgraph(websocket: WebSocket):
    """WebSocket pour interactions temps réel avec l'agent LangGraph
    
    Format d'entrée attendu: {"input_data": "votre question ici"}
    Format de sortie: {"result": "réponse de l'agent"}
    """
    logging.info("Nouvelle connexion WebSocket reçue sur /v1/agents/langgraph/ws")
    
    try:
        await websocket.accept()
        logging.info("Connexion WebSocket acceptée avec succès")
        
        # Générer un ID utilisateur unique s'il n'est pas fourni
        user_id = websocket.query_params.get('user_id', str(uuid.uuid4()))
        logging.info(f"Connexion établie pour l'utilisateur: {user_id}")
        
        # Configuration pour identifier la session
        config = {"configurable": {"thread_id": f"user-{user_id}"}}
        
        # Vérifier si c'est un nouvel utilisateur
        is_first_interaction = False
        if user_id not in user_states:
            is_first_interaction = True
            user_states[user_id] = {}
            await handle_welcome(user_id, websocket)
        
        # Boucle principale de traitement des messages
        while True:
            # Recevoir le message
            data = await websocket.receive_text()
            input_data = json.loads(data)
            logging.info(f"Données reçues: {input_data}")
            
            # Vérifier que les données essentielles sont présentes
            if 'input_data' not in input_data:
                logging.error("Clé manquante dans le payload: 'input_data'")
                await websocket.send_text(json.dumps({
                    'status': 'error', 
                    'message': 'Format invalide: la clé "input_data" est requise'
                }))
                continue
            
            # Sauvegarder l'entrée originale
            original_input = input_data.get('input_data', '')
            if user_id not in user_states:
                user_states[user_id] = {}
            user_states[user_id]['original_input'] = original_input
            
            # Déterminer le type de requête
            request_type = determine_request_type(user_id, input_data)
            
            # Traiter selon le type de requête
            if request_type == "JOKE_MODE":
                await handle_joke_mode(user_id, input_data, websocket, config)
            elif request_type == "EXIT_JOKE_MODE":
                await handle_exit_joke_mode(user_id, websocket)
            else:  # STANDARD_REQUEST
                await handle_standard_request(user_id, input_data, websocket, config, is_first_interaction)
                # Réinitialiser le flag après la première interaction
                if is_first_interaction:
                    is_first_interaction = False
            
    except WebSocketDisconnect:
        logging.info(f"WebSocket déconnecté pour l'utilisateur {user_id}")
    except Exception as e:
        logging.error(f"Erreur dans websocket_langgraph: {str(e)}")
        traceback.print_exc()
        try:
            await websocket.send_text(json.dumps({
                "error": f"Une erreur s'est produite: {str(e)}"
            }))
        except:
            logging.error("Impossible d'envoyer le message d'erreur, la connexion est peut-être fermée")
