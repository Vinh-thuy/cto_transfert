from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict

app = FastAPI()

class Message(BaseModel):
    sender: str
    content: str
    timestamp: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_names: Dict[WebSocket, str] = {}

    async def connect(self, websocket: WebSocket, client_name: str = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if client_name:
            self.client_names[websocket] = client_name

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        if websocket in self.client_names:
            del self.client_names[websocket]

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_name}")
async def websocket_endpoint(websocket: WebSocket, client_name: str):
    await manager.connect(websocket, client_name)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"{client_name}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"{client_name} a quitté la conversation")

@app.post("/send_message")
async def send_message(message: Message):
    """Endpoint REST pour envoyer un message"""
    await manager.broadcast(f"{message.sender}: {message.content}")
    return {"status": "Message envoyé"}

@app.get("/active_connections")
async def get_active_connections():
    """Endpoint pour récupérer les connexions actives"""
    return {
        "total_connections": len(manager.active_connections),
        "clients": list(manager.client_names.values())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
