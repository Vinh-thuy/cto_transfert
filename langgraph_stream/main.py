"""
Point d'entrée principal de l'application
"""

import sys
import os
from api.websocket_server import start_server

if __name__ == "__main__":
    # Ajout du chemin courant au PYTHONPATH
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Démarrage du serveur WebSocket
    start_server()
