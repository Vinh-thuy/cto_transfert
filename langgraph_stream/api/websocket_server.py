"""
Serveur WebSocket pour l'API
"""

import asyncio
import json
import logging
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import sys

# Ajout du chemin parent au PYTHONPATH pour pouvoir importer le module agent
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import execute_graph_stream

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangGraph Streaming API")

# Configuration CORS pour permettre les requêtes du frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifiez les origines exactes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chemin vers le répertoire app
app_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")

# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory=app_dir), name="static")

class ConnectionManager:
    """Gestionnaire de connexions WebSocket"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection established. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket connection closed. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message, websocket, agent_id='default'):
        """
        Envoyer un message avec support des différents types de données
        
        Args:
            message: Peut être un message simple, un dictionnaire d'état de graphe, ou une liste de messages
            websocket: Le websocket de destination
            agent_id: Identifiant de l'agent par défaut
        """
        try:
            # Gestion des états de graphe LangGraph
            if isinstance(message, dict) and 'messages' in message:
                for msg in message.get('messages', []):
                    await websocket.send_json({
                        'type': 'stream',
                        'content': msg.get('content', ''),
                        'agent_id': msg.get('agent_id', agent_id)
                    })
            
            # Gestion des messages simples
            elif isinstance(message, str):
                await websocket.send_json({
                    'type': 'stream',
                    'content': message,
                    'agent_id': agent_id
                })
            
            # Gestion des listes de messages
            elif isinstance(message, list):
                for item in message:
                    await websocket.send_json({
                        'type': 'stream',
                        'content': str(item),
                        'agent_id': agent_id
                    })
            
            # Gestion des autres types
            else:
                await websocket.send_json({
                    'type': 'stream',
                    'content': str(message),
                    'agent_id': agent_id
                })
        
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    async def stream_response(self, websocket, graph_state):
        """Streamer les réponses avec support multi-agents"""
        for message in graph_state['messages']:
            # Envoyer l'identifiant de l'agent avec le message
            await websocket.send_json({
                'type': 'stream',
                'content': message['content'],
                'agent_id': message.get('agent_id', 'default')
            })

manager = ConnectionManager()

@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Attente d'un message du client
            try:
                data = await websocket.receive_text()
                logger.info(f"Received message: {data}")
            except WebSocketDisconnect:
                manager.disconnect(websocket)
                break
            
            # Exécution du workflow LangGraph en streaming
            try:
                # Premier message (Salutation)
                await manager.send_message("[START_MESSAGE]", websocket)
                async for chunk in execute_graph_stream(human_input=data):
                    # Envoi de chaque morceau au client en temps réel
                    logger.info(f"Sending chunk: {chunk}")
                    await manager.send_message(chunk, websocket)
                await manager.send_message("[END_MESSAGE]", websocket)
                
                # Attente d'une seconde entre les messages
                await asyncio.sleep(1)
                
                # Note: Avec le nouveau workflow LangGraph, nous n'avons plus besoin d'un deuxième message
                # car la conversation est gérée par le graphe lui-même
                
                # Signaler la fin du flux
                await manager.send_message("[END]", websocket)
            
            except Exception as e:
                logger.error(f"Error in workflow execution: {e}")
                await manager.send_message(f"Error: {str(e)}", websocket)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)

@app.get("/")
async def root():
    """Servir le fichier index.html"""
    return FileResponse(os.path.join(app_dir, "index.html"))

def start_server():
    """Démarre le serveur uvicorn"""
    uvicorn.run("api.websocket_server:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start_server()
