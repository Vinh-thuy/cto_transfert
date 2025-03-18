from fastapi import APIRouter, WebSocket
import json
import logging
from pydantic import BaseModel
from agents.LangGraph_agent import router_workflow  # Import absolu depuis la racine

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
                
                logging.info(f"Appel de router_workflow.invoke avec {workflow_input}")
                try:
                    state = router_workflow.invoke(workflow_input)
                    logging.info(f"Résultat obtenu: {state}")
                    
                    if "output" not in state:
                        logging.error(f"Clé 'output' manquante dans l'état: {state}")
                        response = {"error": "Erreur interne: format de réponse invalide", "details": str(state)}
                    else:
                        # Format de réponse cohérent avec l'API REST
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
