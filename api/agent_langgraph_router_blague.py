from fastapi import APIRouter, WebSocket
import json
import logging
from pydantic import BaseModel
from agents.LangGraph_agent import router_workflow, joke_workflow  # Import absolu depuis la racine

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
                
                # Log des données reçues pour débogage
                logging.info(f"Données reçues: {input_data}")
                if 'input_data' not in input_data or 'user_id' not in input_data:
                    logging.error("Clé manquante dans le payload: 'input_data' ou 'user_id'")
                    return {'status': 'error', 'message': 'Clé manquante: None'}
                
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
                
                # Log pour vérifier si nous avons un user_id valide
                logging.info(f"Vérification de user_id: {user_id}")
                if user_id not in user_states:
                    logging.info(f"user_id {user_id} non trouvé, ajout d'un nouvel utilisateur.")
                else:
                    logging.info(f"user_id {user_id} trouvé, état actuel: {user_states[user_id]}")
                    
                
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
                
                # Créer l'état de l'utilisateur
                state = {
                    'input': input_data.get('input_data', ''),
                    'decision': 'initial_decision',
                    'output': '',
                    'user_approval': False,
                    'conversation_ended': False,
                    'in_conversation': True,
                    'current_region': 'suisse',  # ou une autre valeur selon le contexte
                    'agent_name': 'Multi-Region RAG'  # ou une autre valeur selon le contexte
                }

                # Log de l'état avant l'invocation
                logging.info(f"État de l'utilisateur avant invocation: {state}")

                # Appeler le workflow avec l'état
                workflow_input = {'input': input_data.get('input_data'), 'state': state}
                result = router_workflow.invoke(workflow_input)

                # Log de la réponse du workflow
                logging.info(f"Réponse du workflow: {result}")
                
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
                        logging.info(f"Mise à jour de awaiting_validation pour {user_id}: False")  # Log pour mise à jour
                        
                        # Contourner le routage normal pour cette réponse de validation
                        logging.info("Contournement du routage normal pour la réponse de validation")
                        logging.info(f"Préparation de l'input pour continuer le flux: {workflow_input}")
                    elif awaiting_validation:
                        # Si nous attendons une validation mais que la réponse n'est pas reconnue
                        logging.info("Réponse non reconnue comme validation")
                        if isinstance(input_data.get('input_data'), str):
                            original_input = input_data.get('input_data')
                            # Sauvegarder l'entrée originale pour une utilisation ultérieure
                            if user_id not in user_states:
                                user_states[user_id] = {}
                                logging.info(f"Nouvel utilisateur ajouté: {user_id}")  # Log pour nouvel utilisateur
                            # Log pour vérifier l'état de l'utilisateur avant la mise à jour
                            logging.info(f"État de user_states avant mise à jour pour {user_id}: {user_states.get(user_id, 'Non trouvé')}")
                            user_states[user_id]['original_input'] = original_input
                            logging.info(f"Mise à jour de original_input pour {user_id}: {original_input}")  # Log pour mise à jour
                            logging.info(f"État de user_states après mise à jour pour {user_id}: {user_states[user_id]}")
                            # Réinitialiser l'état d'attente car l'utilisateur a envoyé une nouvelle requête
                            user_states[user_id]['awaiting_validation'] = False
                            logging.info(f"Mise à jour de awaiting_validation pour {user_id}: False")  # Log pour mise à jour
                            workflow_input = {'input': original_input}
                    else:
                        # Si nous n'attendons pas de validation, traiter comme une nouvelle requête
                        if isinstance(input_data.get('input_data'), str):
                            original_input = input_data.get('input_data')
                            # Sauvegarder l'entrée originale pour une utilisation ultérieure
                            if user_id not in user_states:
                                user_states[user_id] = {}
                                logging.info(f"Nouvel utilisateur ajouté: {user_id}")  # Log pour nouvel utilisateur
                            user_states[user_id]['original_input'] = original_input
                            logging.info(f"Mise à jour de original_input pour {user_id}: {original_input}")  # Log pour mise à jour
                            logging.info(f"État de user_states après mise à jour pour {user_id}: {user_states[user_id]}")
                            workflow_input = {'input': original_input}
                    
                    # Vérifier si l'utilisateur attend de fournir un thème pour la blague
                    if user_id in user_states and user_states[user_id].get('waiting_for_theme', False):
                        # L'utilisateur a fourni un thème pour la blague
                        theme = original_input
                        logging.info(f"Thème de blague reçu de l'utilisateur {user_id}: {theme}")
                        
                        # Construire une prompt personnalisée avec le thème
                        joke_prompt = f"Raconte-moi une blague sur le thème: {theme}. Fais en sorte qu'elle soit drôle et adaptée à tous les publics."
                        logging.info(f"Prompt personnalisé pour la blague: {joke_prompt}")
                        
                        # Appeler directement le workflow de blagues avec le thème
                        state = joke_workflow.invoke({"input": joke_prompt}, config=config)
                        logging.info(f"Blague générée sur le thème '{theme}': {state}")
                        
                        # Préparer la réponse avec la blague générée
                        if "output" in state:
                            response = {
                                "result": state["output"],
                                "agent_name": "Assistant Blagues",
                                "theme": theme
                            }
                            logging.info(f"Envoi de la blague sur le thème '{theme}': {response['result']}")
                            
                            # Envoyer la réponse au client
                            await websocket.send_text(json.dumps(response))
                            logging.info("Blague envoyée avec succès")
                        else:
                            logging.error(f"Pas de blague générée dans l'état: {state}")
                            await websocket.send_text(json.dumps({"error": "Impossible de générer une blague sur ce thème"}))
                        
                        # Réinitialiser l'attente de thème
                        user_states[user_id]['waiting_for_theme'] = False
                        
                        # Continuer à attendre d'autres messages de l'utilisateur
                        continue
                    else:
                        # Invoquer le workflow normal avec la configuration de session
                        logging.info(f"Appel de router_workflow.invoke avec {workflow_input} (config: {config})")
                        state = router_workflow.invoke(workflow_input, config=config)
                        logging.info(f"Résultat obtenu: {state}")
                    
                    if "output" not in state:
                        logging.error(f"Clé 'output' manquante dans l'état: {state}")
                        response = {"error": "Erreur interne: format de réponse invalide", "details": str(state)}
                    else:
                        # Ajout de l'agent_name à l'état
                        state['agent_name'] = user_states[user_id].get('agent_name', 'Unknown')
                        
                        # Vérifier si nous sommes dans un état qui nécessite une validation
                        if "requires_validation" in state and state["requires_validation"] is True:
                            # Marquer l'utilisateur comme en attente de validation
                            if user_id not in user_states:
                                user_states[user_id] = {}
                                logging.info(f"Nouvel utilisateur ajouté: {user_id}")  # Log pour nouvel utilisateur
                            user_states[user_id]['awaiting_validation'] = True
                            
                            # Stocker la décision originale pour la réutiliser après validation
                            if "decision" in state:
                                user_states[user_id]['original_decision'] = state["decision"]
                                logging.info(f"Décision originale stockée: {state['decision']}")
                            
                            # Format de réponse pour l'interruption
                            response = {
                                "result": state["output"],
                                "agent_name": state['agent_name'],  # Inclure agent_name dans la réponse
                                "waiting_for_approval": True,
                                "message": "Veuillez confirmer pour continuer"
                            }
                            logging.info("Interruption détectée, attente de validation utilisateur")
                        else:
                            # Format de réponse standard
                            # Log pour déboguer le contenu exact de la sortie
                            logging.info(f"Contenu brut de state['output']: {state['output']}")
                            
                            # Vérifier si c'est une réponse du noeud joke_intro
                            if state.get("decision") == "joke_intro" or state.get("decision") == "joke":
                                # Envoyer le message d'introduction qui demande le thème
                                response = {
                                    "result": state["output"],
                                    "agent_name": state.get('agent_name', 'Assistant Blagues'),
                                    "waiting_for_theme": True  # Indiquer qu'on attend un thème pour la blague
                                }
                                
                                # Stocker l'information que l'utilisateur doit fournir un thème
                                if user_id not in user_states:
                                    user_states[user_id] = {}
                                user_states[user_id]['waiting_for_theme'] = True
                                user_states[user_id]['original_joke_request'] = state.get("original_input", state["input"])
                                logging.info(f"Utilisateur {user_id} en attente de thème pour la blague. Requête originale: {user_states[user_id]['original_joke_request']}")
                            else:
                                # Réponse standard pour les autres noeuds
                                response = {
                                    "result": state["output"],
                                    "agent_name": state['agent_name']  # Inclure agent_name dans la réponse
                                }
                    
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
