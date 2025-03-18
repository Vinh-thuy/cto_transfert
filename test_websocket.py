#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script de test pour vérifier la connexion WebSocket avec l'agent LangGraph.
Ce script se connecte au WebSocket, envoie un message et affiche la réponse.
"""

import asyncio
import json
import websockets
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Configuration du WebSocket
WEBSOCKET_HOST = 'localhost'
WEBSOCKET_PORT = 8002
WEBSOCKET_PATH = '/v1/agents/langgraph/ws'
USER_ID = 'test_user'

# URI du WebSocket
WEBSOCKET_URI = f"ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}{WEBSOCKET_PATH}?user_id={USER_ID}"


async def test_websocket_connection():
    """Teste la connexion WebSocket avec l'agent LangGraph."""
    try:
        logging.info(f"Tentative de connexion à : {WEBSOCKET_URI}")
        
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            logging.info("✅ Connexion WebSocket établie")
            
            # Préparer le message à envoyer
            message = {
                "input": "Raconte-moi une blague sur les chats"
            }
            
            # Convertir en JSON et envoyer
            json_message = json.dumps(message)
            logging.info(f"📤 Envoi du message : {json_message}")
            await websocket.send(json_message)
            logging.info("📩 Message envoyé avec succès")
            
            # Attendre la réponse
            logging.info("⏳ En attente de réponse...")
            response = await websocket.recv()
            logging.info(f"📥 Réponse reçue : {response}")
            
            # Analyser la réponse
            try:
                parsed_response = json.loads(response)
                logging.info(f"🔍 Réponse analysée : {json.dumps(parsed_response, indent=2, ensure_ascii=False)}")
                
                # Vérifier si la réponse contient une erreur
                if "error" in parsed_response:
                    logging.error(f"❌ Erreur reçue : {parsed_response['error']}")
                else:
                    logging.info("✅ Communication réussie avec l'agent LangGraph")
            except json.JSONDecodeError as e:
                logging.error(f"❌ Erreur lors de l'analyse de la réponse JSON : {e}")
                logging.error(f"Réponse brute : {response}")
    
    except websockets.exceptions.ConnectionClosedError as e:
        logging.error(f"❌ La connexion a été fermée de manière inattendue : {e}")
    except Exception as e:
        logging.error(f"❌ Erreur lors de la connexion WebSocket : {type(e).__name__} - {e}")


async def test_multiple_messages():
    """Teste l'envoi de plusieurs messages consécutifs."""
    try:
        logging.info(f"Tentative de connexion à : {WEBSOCKET_URI}")
        
        async with websockets.connect(WEBSOCKET_URI) as websocket:
            logging.info("✅ Connexion WebSocket établie")
            
            # Liste de messages à tester
            messages = [
                "Raconte-moi une blague",
                "Écris un poème sur la nature",
                "Raconte-moi une histoire courte"
            ]
            
            for msg in messages:
                # Préparer le message
                message = {"input": msg}
                json_message = json.dumps(message)
                
                # Envoyer le message
                logging.info(f"📤 Envoi du message : {json_message}")
                await websocket.send(json_message)
                logging.info("📩 Message envoyé avec succès")
                
                # Attendre la réponse
                logging.info("⏳ En attente de réponse...")
                response = await websocket.recv()
                logging.info(f"📥 Réponse reçue pour '{msg}'")
                
                # Analyser la réponse
                try:
                    parsed_response = json.loads(response)
                    if "response" in parsed_response:
                        content = parsed_response["response"]
                        # Afficher les 100 premiers caractères de la réponse
                        preview = content[:100] + "..." if len(content) > 100 else content
                        logging.info(f"🔍 Réponse : {preview}")
                    else:
                        logging.warning(f"⚠️ Format de réponse inattendu : {parsed_response}")
                except json.JSONDecodeError:
                    logging.error(f"❌ Réponse non-JSON : {response}")
                
                # Attendre un peu avant le prochain message
                await asyncio.sleep(1)
    
    except Exception as e:
        logging.error(f"❌ Erreur : {type(e).__name__} - {e}")


async def main():
    """Fonction principale qui exécute les tests."""
    logging.info("🚀 Démarrage des tests WebSocket pour l'agent LangGraph")
    
    # Test de connexion simple
    logging.info("\n=== Test de connexion simple ===")
    await test_websocket_connection()
    
    # Test avec plusieurs messages
    logging.info("\n=== Test avec plusieurs messages ===")
    await test_multiple_messages()
    
    logging.info("✅ Tests terminés")


if __name__ == "__main__":
    # Exécuter la fonction principale de manière asynchrone
    asyncio.run(main())
