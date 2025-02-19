import os
import asyncio
import panel as pn
import websockets
import json
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

class ChatInterface:
    def __init__(self):
        self.websocket = None
        self.chat_area = pn.Column(sizing_mode="stretch_width")
        self.input_box = pn.TextInput(placeholder="Entrez votre message...", sizing_mode="stretch_width")
        self.send_button = pn.Button(name="Envoyer", button_type="primary")
        
        # Configuration des événements
        self.send_button.on_click(self.send_message)
        self.input_box.param.watch(self.on_input_change, 'value')
        
        # Layout
        self.layout = pn.Column(
            pn.pane.Markdown("# 💬 Chat IA"),
            self.chat_area,
            pn.Row(self.input_box, self.send_button),
            sizing_mode="stretch_width"
        )
    
    async def connect_websocket(self):
        """Établir la connexion WebSocket"""
        try:
            self.websocket = await websockets.connect("ws://localhost:8000/ws/chat")
            self.add_message("Système", "Connexion établie avec succès !", "info")
        except Exception as e:
            self.add_message("Système", f"Erreur de connexion : {e}", "error")
    
    def add_message(self, sender, message, message_type="message"):
        """Ajouter un message à l'interface"""
        color = {
            "message": "black",
            "info": "blue",
            "error": "red"
        }.get(message_type, "black")
        
        message_pane = pn.pane.Markdown(
            f"**{sender}**: {message}",
            style={"color": color},
            sizing_mode="stretch_width"
        )
        self.chat_area.append(message_pane)
    
    async def receive_messages(self):
        """Recevoir des messages du serveur WebSocket"""
        while True:
            try:
                message = await self.websocket.recv()
                self.add_message("Assistant", message)
            except websockets.exceptions.ConnectionClosed:
                self.add_message("Système", "Connexion perdue", "error")
                break
    
    def send_message(self, event=None):
        """Envoyer un message via WebSocket"""
        message = self.input_box.value
        if message and self.websocket:
            asyncio.create_task(self._async_send(message))
            self.add_message("Vous", message)
            self.input_box.value = ""  # Réinitialiser l'input
    
    async def _async_send(self, message):
        """Envoi asynchrone du message"""
        await self.websocket.send(message)
    
    def on_input_change(self, event):
        """Gérer la touche Entrée dans l'input"""
        if event.new and event.new.strip():
            self.send_button.disabled = False
        else:
            self.send_button.disabled = True
    
    def start(self):
        """Démarrer l'interface de chat"""
        async def main():
            await self.connect_websocket()
            receive_task = asyncio.create_task(self.receive_messages())
            await receive_task
        
        pn.extension(comms='vscode')
        asyncio.run(main())
        return self.layout

# Lancement de l'application
def main():
    chat_interface = ChatInterface()
    app = chat_interface.start()
    app.servable()
    pn.serve(app)

if __name__ == "__main__":
    main()
