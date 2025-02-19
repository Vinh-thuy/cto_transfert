import panel as pn
import asyncio
import websockets
import json
from typing import List

class ChatApplication:
    def __init__(self, websocket_url: str = "ws://localhost:8765"):
        pn.extension()
        
        self.websocket_url = websocket_url
        self.messages: List[str] = []
        
        # Éléments d'interface
        self.message_box = pn.Column(sizing_mode='stretch_width')
        self.input = pn.TextInput(placeholder='Entrez votre message...', sizing_mode='stretch_width')
        self.send_button = pn.Button(name='Envoyer', button_type='primary')
        
        # Configuration des événements
        self.send_button.on_click(self.send_message)
        self.input.bind(self.send_message, pn.Param('value', precedence=-1))
        
        # Initialisation de la connexion WebSocket
        self.websocket = None
        
    async def connect_websocket(self):
        """Établir la connexion WebSocket"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            await self.receive_messages()
        except Exception as e:
            self.message_box.append(f"Erreur de connexion : {e}")
    
    async def send_message(self, event=None):
        """Envoyer un message via WebSocket"""
        message = self.input.value.strip()
        if message and self.websocket:
            try:
                await self.websocket.send(json.dumps({
                    'type': 'chat',
                    'message': message
                }))
                self.input.value = ''  # Réinitialiser l'input
            except Exception as e:
                self.message_box.append(f"Erreur d'envoi : {e}")
    
    async def receive_messages(self):
        """Recevoir et afficher les messages"""
        try:
            async for raw_message in self.websocket:
                message = json.loads(raw_message)
                if message.get('type') == 'chat':
                    self.message_box.append(f"{message.get('sender', 'Anonyme')}: {message.get('message', '')}")
        except websockets.exceptions.ConnectionClosed:
            self.message_box.append("Connexion WebSocket fermée")
    
    def create_layout(self):
        """Créer la mise en page de l'application"""
        layout = pn.Column(
            pn.pane.Markdown("# Chat WebSocket"),
            self.message_box,
            pn.Row(self.input, self.send_button),
            sizing_mode='stretch_width'
        )
        return layout
    
    def run(self):
        """Démarrer l'application"""
        layout = self.create_layout()
        
        # Démarrer la connexion WebSocket de manière asynchrone
        async def start():
            await self.connect_websocket()
        
        asyncio.get_event_loop().create_task(start())
        
        return layout.servable()

# Démarrer l'application
if __name__ == '__main__':
    chat_app = ChatApplication()
    pn.serve(chat_app.run())
