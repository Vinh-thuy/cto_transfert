import asyncio
import json
import websockets
from typing import Dict, Any

class WebSocketServer:
    def __init__(self, host: str = 'localhost', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Dict[websockets.WebSocketServerProtocol, str] = {}

    async def register(self, websocket: websockets.WebSocketServerProtocol):
        """Enregistrer un nouveau client WebSocket"""
        client_id = f"client_{len(self.clients) + 1}"
        self.clients[websocket] = client_id
        return client_id

    async def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Désenregistrer un client WebSocket"""
        if websocket in self.clients:
            del self.clients[websocket]

    async def broadcast(self, message: str, sender: websockets.WebSocketServerProtocol = None):
        """Diffuser un message à tous les clients sauf l'expéditeur"""
        if not self.clients:
            return
        
        for client in self.clients:
            if client != sender:
                try:
                    await client.send(message)
                except websockets.exceptions.ConnectionClosedError:
                    await self.unregister(client)

    async def handle_message(self, websocket: websockets.WebSocketServerProtocol, message: str):
        """Gérer les messages entrants"""
        try:
            data = json.loads(message)
            client_id = self.clients.get(websocket, "Unknown")
            
            # Exemple de traitement de message
            if data.get('type') == 'chat':
                broadcast_message = json.dumps({
                    'type': 'chat',
                    'sender': client_id,
                    'message': data.get('message', '')
                })
                await self.broadcast(broadcast_message, websocket)
            
            elif data.get('type') == 'system':
                # Gestion des messages système
                print(f"System message from {client_id}: {data}")
        
        except json.JSONDecodeError:
            print(f"Invalid JSON: {message}")
        except Exception as e:
            print(f"Error handling message: {e}")

    async def handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Gestionnaire principal des connexions WebSocket"""
        client_id = await self.register(websocket)
        
        try:
            # Notification de connexion
            await websocket.send(json.dumps({
                'type': 'system',
                'message': f'Connecté avec l\'ID {client_id}'
            }))
            
            async for message in websocket:
                await self.handle_message(websocket, message)
        
        except websockets.exceptions.ConnectionClosed:
            print(f"Client {client_id} déconnecté")
        finally:
            await self.unregister(websocket)

    def start(self):
        """Démarrer le serveur WebSocket"""
        server = websockets.serve(self.handler, self.host, self.port)
        print(f"Serveur WebSocket démarré sur ws://{self.host}:{self.port}")
        
        asyncio.get_event_loop().run_until_complete(server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    ws_server = WebSocketServer()
    ws_server.start()
