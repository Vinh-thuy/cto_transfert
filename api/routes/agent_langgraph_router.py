from fastapi import APIRouter, WebSocket
import json
import logging
from pydantic import BaseModel
from agents.LangGraph_agent import router_workflow  # Import absolu depuis la racine

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

@router.websocket("/langgraph/ws")
async def websocket_langgraph(websocket: WebSocket):
    """WebSocket pour interactions temps réel avec l'agent LangGraph
    
    Format d'entrée attendu: {"input_data": "votre question ici"}
    Format de sortie: {"result": "réponse de l'agent"}
    """
    logging.info("Nouvelle connexion WebSocket reçue sur /v1/agents/langgraph/ws")
    logging.info(f"Headers de connexion: {websocket.headers}")
    logging.info(f"Paramètres de requête: {websocket.query_params}")
    
    try:
        await websocket.accept()
        logging.info("Connexion WebSocket acceptée avec succès")
    except Exception as e:
        logging.error(f"Erreur lors de l'acceptation de la connexion WebSocket: {type(e).__name__} - {str(e)}")
        return
    try:
        while True:
            logging.info("En attente de données du client...")
            data = await websocket.receive_text()
            logging.info(f"Données reçues: {data}")
            
            try:
                input_data = json.loads(data)
                logging.info(f"Données JSON parsées: {input_data}")
                
                # Priorité au format simplifié avec input_data
                workflow_input = {}
                
                if isinstance(input_data, str):
                    # Si c'est une chaîne, la traiter comme entrée directe
                    logging.info(f"Format de chaîne reçu, traité comme entrée directe: {input_data}")
                    workflow_input = {'input': input_data}
                elif 'input_data' in input_data:
                    # Format simplifié recommandé
                    logging.info(f"Format simplifié détecté avec input_data: {input_data['input_data']}")
                    workflow_input = {'input': input_data['input_data']}
                elif 'input' in input_data:
                    # Format compatible avec le workflow directement
                    logging.info(f"Format compatible avec le workflow détecté: {input_data}")
                    workflow_input = input_data
                else:
                    # Essayer de détecter d'autres formats courants
                    logging.warning(f"Format non standard détecté: {input_data}")
                    for key in ['query', 'message', 'text', 'content', 'question']:
                        if key in input_data:
                            workflow_input = {'input': input_data.get(key)}
                            logging.info(f"Format converti depuis la clé '{key}': {workflow_input}")
                            break
                    else:
                        # Si aucun format reconnu, utiliser une valeur par défaut
                        workflow_input = {'input': 'Bonjour'}
                        logging.warning(f"Aucun format reconnu, utilisation de la valeur par défaut: {workflow_input}")
                
                # Extraire l'ID utilisateur pour la gestion des sessions
                user_id = input_data.get('user_id', websocket.query_params.get('user_id', 'anonymous'))
                
                # Vérifier si l'utilisateur a un état en attente de validation
                awaiting_validation = user_id in user_states and user_states[user_id].get('awaiting_validation', False)
                
                # Variable pour suivre si nous avons une réponse de validation
                user_approval = None
                
                # Si nous attendons une validation, vérifier si cette entrée est une réponse de validation
                if awaiting_validation:
                    logging.info(f"Utilisateur {user_id} en attente de validation. Vérification de la réponse...")
                    
                    # Vérifier si la réponse est explicitement marquée comme validation
                    if 'user_approval' in input_data:
                        user_approval = input_data['user_approval']
                    # Ensuite, vérifier si la réponse est dans le champ 'response'
                    elif 'response' in input_data:
                        # Traiter les réponses textuelles comme oui/non
                        response_text = input_data.get('response', '').lower().strip()
                        if response_text in ['oui', 'yes', 'y', 'ok', 'continue', 'continuer']:
                            user_approval = True
                        elif response_text in ['non', 'no', 'n', 'cancel', 'annuler']:
                            user_approval = False
                    # Enfin, vérifier si l'input_data lui-même est une réponse oui/non
                    elif isinstance(input_data.get('input_data'), str):
                        response_text = input_data.get('input_data', '').lower().strip()
                        if response_text in ['oui', 'yes', 'y', 'ok', 'continue', 'continuer']:
                            user_approval = True
                            logging.info(f"Détection d'une réponse positive: {response_text}")
                        elif response_text in ['non', 'no', 'n', 'cancel', 'annuler']:
                            user_approval = False
                            logging.info(f"Détection d'une réponse négative: {response_text}")
                
                # Configuration pour identifier la session
                config = {"configurable": {"thread_id": f"user-{user_id}"}}
                
                # Préparer l'entrée pour le workflow
                try:
                    # Initialiser workflow_input avec une valeur par défaut
                    workflow_input = {'input': ''}
                    
                    # Si nous avons une réponse de validation et que l'utilisateur est en attente de validation
                    if awaiting_validation and user_approval is not None:
                        # Récupérer l'entrée originale et la décision qui ont déclenché la demande de validation
                        original_input = user_states[user_id].get('original_input', 'tell me a story')
                        original_decision = user_states[user_id].get('original_decision', 'story')
                        
                        logging.info(f"Utilisation de l'entrée originale: {original_input} avec décision: {original_decision}")
                        
                        # IMPORTANT: Pour les réponses de validation, nous ne voulons pas que le routeur LLM interprète
                        # la réponse 'oui'/'non', mais plutôt utiliser directement l'état précédent avec la décision originale
                        # et l'approbation utilisateur
                        
                        # Créer un état spécial qui sera traité directement par check_user_approval
                        # IMPORTANT: Pour le cas 'story', nous forçons la décision à 'story' pour éviter tout problème de routage
                        workflow_input = {
                            # Nous utilisons l'entrée originale pour conserver le contexte
                            'input': original_input,
                            'user_approval': user_approval,
                            'decision': 'story' if original_decision == 'story' else original_decision,
                            # Ajouter un flag spécial pour contourner le routeur LLM
                            'bypass_router': True,
                            # Ajouter un flag pour indiquer que c'est une réponse de validation
                            'is_validation_response': True
                        }
                        
                        # Journalisation détaillée
                        logging.info(f"Réponse de validation détectée: {user_approval}")
                        logging.info(f"Décision originale: {original_decision}")
                        
                        # Réinitialiser l'état d'attente
                        user_states[user_id]['awaiting_validation'] = False
                        
                        # Contourner le routage normal pour cette réponse de validation
                        logging.info("Contournement du routage normal pour la réponse de validation")
                        logging.info(f"Préparation de l'input pour continuer le flux: {workflow_input}")
                    elif awaiting_validation:
                        # Si nous attendons une validation mais que la réponse n'est pas reconnue
                        logging.info("Réponse non reconnue comme validation")
                        if isinstance(input_data.get('input_data'), str):
                            original_input = input_data.get('input_data')
                            # Sauvegarder l'entrée originale pour une utilisation ultérieure
                            user_states[user_id]['original_input'] = original_input
                            # Réinitialiser l'état d'attente car l'utilisateur a envoyé une nouvelle requête
                            user_states[user_id]['awaiting_validation'] = False
                            workflow_input = {'input': original_input}
                    else:
                        # Si nous n'attendons pas de validation, traiter comme une nouvelle requête
                        if isinstance(input_data.get('input_data'), str):
                            original_input = input_data.get('input_data')
                            # Sauvegarder l'entrée originale pour une utilisation ultérieure
                            if user_id not in user_states:
                                user_states[user_id] = {}
                            user_states[user_id]['original_input'] = original_input
                            workflow_input = {'input': original_input}
                    
                    # Invoquer le workflow avec la configuration de session
                    logging.info(f"Appel de router_workflow.invoke avec {workflow_input} (config: {config})")
                    state = router_workflow.invoke(workflow_input, config=config)
                    logging.info(f"Résultat obtenu: {state}")
                    
                    if "output" not in state:
                        logging.error(f"Clé 'output' manquante dans l'état: {state}")
                        response = {"error": "Erreur interne: format de réponse invalide", "details": str(state)}
                    else:
                        # Vérifier si nous sommes dans un état qui nécessite une validation
                        if "requires_validation" in state and state["requires_validation"] is True:
                            # Marquer l'utilisateur comme en attente de validation
                            if user_id not in user_states:
                                user_states[user_id] = {}
                            user_states[user_id]['awaiting_validation'] = True
                            
                            # Stocker la décision originale pour la réutiliser après validation
                            if "decision" in state:
                                user_states[user_id]['original_decision'] = state["decision"]
                                logging.info(f"Décision originale stockée: {state['decision']}")
                            
                            # Format de réponse pour l'interruption
                            response = {
                                "result": state["output"],
                                "waiting_for_approval": True,
                                "message": "Veuillez confirmer pour continuer"
                            }
                            logging.info("Interruption détectée, attente de validation utilisateur")
                        else:
                            # Format de réponse standard
                            response = {"result": state["output"]}
                    
                    logging.info(f"Envoi de la réponse: {response}")
                    json_response = json.dumps(response)
                    logging.info(f"JSON de réponse: {json_response}")
                    await websocket.send_text(json_response)
                    logging.info("Réponse envoyée avec succès")
                except Exception as e:
                    logging.error(f"Erreur lors de l'invocation du workflow: {type(e).__name__} - {str(e)}")
                    error_response = {"error": f"Erreur lors du traitement: {type(e).__name__}", "details": str(e)}
                    await websocket.send_text(json.dumps(error_response))
            except json.JSONDecodeError as je:
                logging.error(f"Erreur de décodage JSON: {je}")
                await websocket.send_text(json.dumps({"error": "Format JSON invalide"}))
            except KeyError as ke:
                logging.error(f"Clé manquante: {ke}")
                await websocket.send_text(json.dumps({"error": f"Clé manquante: {ke}"}))    
    except Exception as e:
        logging.error(f"Exception dans le WebSocket: {type(e).__name__} - {str(e)}")
        try:
            # Essayer d'envoyer un message d'erreur avant de fermer
            error_message = {"error": f"Exception WebSocket: {type(e).__name__}", "details": str(e)}
            await websocket.send_text(json.dumps(error_message))
            logging.info("Message d'erreur envoyé avant fermeture")
        except Exception as send_error:
            logging.error(f"Impossible d'envoyer le message d'erreur: {send_error}")
        finally:
            # Fermer la connexion avec un code d'erreur
            await websocket.close(code=1011, reason=str(e))
