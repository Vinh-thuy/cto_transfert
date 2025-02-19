
# 2️⃣ 🎯 Objectif : Adapter le code pour un chatbot

# Nous devons modifier pre_hook pour que :
# 	1.	L’agent envoie une demande de confirmation au chatbot.
# 	2.	Le chatbot affiche une boîte de dialogue interactive (ex: bouton “Confirmer”).
# 	3.	L’utilisateur confirme/refuse via l’interface Panel.
# 	4.	Le backend récupère la réponse et continue ou stoppe l’exécution.


# 3️⃣ 🔄 Adaptation avec FastAPI + WebSockets
# Nous allons :
# 	•	Remplacer Prompt.ask() par un WebSocket entre le chatbot et le backend.
# 	•	Utiliser Panel pour afficher des boutons de confirmation.

import asyncio
from fastapi import FastAPI, WebSocket
from agno.agent import Agent
from agno.exceptions import StopAgentRun
from agno.tools import FunctionCall, tool
from rich.console import Console

app = FastAPI()
console = Console()

# Stocker les WebSockets connectés
active_connections = {}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """ WebSocket pour la confirmation utilisateur """
    await websocket.accept()
    active_connections["user"] = websocket  # Associer à un ID si multi-utilisateurs

async def pre_hook(fc: FunctionCall):
    """Envoie une demande de confirmation via WebSocket au lieu de la console"""
    console.print(f"\nWaiting for confirmation for [bold blue]{fc.function.name}[/]")

    # Envoie une requête de confirmation au chatbot via WebSocket
    websocket = active_connections.get("user")
    if websocket:
        await websocket.send_text(f"CONFIRM|{fc.function.name}")

        # Attend la réponse de l'utilisateur
        response = await websocket.receive_text()

        if response.lower() != "y":
            raise StopAgentRun(
                "Tool call cancelled by user",
                agent_message="Stopping execution as permission was not granted.",
            )

# ✅ Ce que fait ce code :
# 	1.	Lorsqu’une action nécessite confirmation, elle envoie un message WebSocket (CONFIRM|Nom de l’action).
# 	2.	Le chatbot affiche un bouton “Confirmer” à l’utilisateur.
# 	3.	L’utilisateur accepte ou refuse via le chatbot.
# 	4.	Le backend lit la réponse et continue ou arrête l’exécution.            



# 📌 2. Interface chatbot avec Panel

# On modifie le chatbot pour :
# 	1.	Écouter les WebSockets pour les confirmations.
# 	2.	Afficher des boutons “Oui / Non”.
# 	3.	Envoyer la réponse au backend.


import panel as pn
import websockets
import asyncio

chatbox = pn.widgets.ChatBox(name="DeepKnowledge Chatbot")
confirm_message = pn.pane.Markdown("En attente de confirmation...", visible=False)
yes_button = pn.widgets.Button(name="✅ Oui", button_type="success", visible=False)
no_button = pn.widgets.Button(name="❌ Non", button_type="danger", visible=False)

async def listen_for_requests():
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        while True:
            message = await websocket.recv()

            if message.startswith("CONFIRM|"):
                action = message.split("|")[1]
                confirm_message.object = f"Confirmer l'exécution de : **{action}**"
                confirm_message.visible = True
                yes_button.visible = True
                no_button.visible = True
                pn.state.param.trigger('confirm')

async def send_confirmation(response: str):
    async with websockets.connect("ws://localhost:8000/ws") as websocket:
        await websocket.send(response)
        confirm_message.visible = False
        yes_button.visible = False
        no_button.visible = False

yes_button.on_click(lambda _: asyncio.create_task(send_confirmation("y")))
no_button.on_click(lambda _: asyncio.create_task(send_confirmation("n")))

layout = pn.Column(
    chatbox,
    confirm_message,
    pn.Row(yes_button, no_button),
)

pn.state.onload(lambda: asyncio.create_task(listen_for_requests()))
layout.show()


# ✅ Ce que fait ce code :
# 	1.	📡 Écoute WebSockets (listen_for_requests()) pour les demandes de confirmation.
# 	2.	📝 Affiche un message et deux boutons (Oui / Non).
# 	3.	🖱️ Envoie la réponse au backend via WebSocket.