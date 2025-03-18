import param
import panel as pn
import asyncio
import json
import websockets
import os
#import threading
import urllib.parse


# Configuration WebSocket
WEBSOCKET_CONFIG = {
    'host': os.getenv('WEBSOCKET_HOST', 'localhost'),  
    'port': int(os.getenv('WEBSOCKET_PORT', 8000)),       # Mise à jour du port pour correspondre au nouveau serveur
    'path': '/v1/agents/langgraph/ws'  # Chemin complet pour l'agent LangGraph
}

model_id: str = 'gpt-4o-mini'
user_id: str = 'vinh'

# Statut de connexion personnalisé
class ConnectionStatusIndicator(pn.widgets.Widget):
    """Indicateur de statut de connexion personnalisé"""
    value = param.Boolean(default=False)
    
    def __init__(self, **params):
        super().__init__(**params)
        self._status_div = pn.pane.HTML(
            self._get_status_html(),
            height=30,
            width=150
        )
    
    def _get_status_html(self):
        """Générer le HTML pour l'indicateur"""
        color = 'green' if self.value else 'red'
        status_text = 'Connecté' if self.value else 'Déconnecté'
        return f"""
        <div style="
            display: flex; 
            align-items: center; 
            padding: 5px; 
            border-radius: 5px; 
            background-color: {color}; 
            color: white;
            font-weight: bold;
        ">
            🔌 {status_text}
        </div>
        """
    
    def _update_status(self, event=None):
        """Mettre à jour l'affichage du statut"""
        self._status_div.object = self._get_status_html()
    
    def __panel__(self):
        """Méthode requise pour les widgets Panel"""
        self.param.watch(self._update_status, 'value')
        return self._status_div
    
    def _get_model(self, doc, comm=None, **kwargs):
        """Implémentation requise pour les widgets Panel"""
        from bokeh.models import Div
        
        color = 'green' if self.value else 'red'
        status_text = 'Connecté' if self.value else 'Déconnecté'
        
        div = Div(text=f"""
        <div style="
            display: flex; 
            align-items: center; 
            padding: 5px; 
            border-radius: 5px; 
            background-color: {color}; 
            color: white;
            font-weight: bold;
        ">
            🔌 {status_text}
        </div>
        """, height=30, width=150)
        
        return div
    
    def _update_model(self, model, old, new):
        """Mettre à jour le modèle Bokeh"""
        color = 'green' if new else 'red'
        status_text = 'Connecté' if new else 'Déconnecté'
        
        model.text = f"""
        <div style="
            display: flex; 
            align-items: center; 
            padding: 5px; 
            border-radius: 5px; 
            background-color: {color}; 
            color: white;
            font-weight: bold;
        ">
            🔌 {status_text}
        </div>
        """

# Créer l'indicateur de connexion
connection_status = ConnectionStatusIndicator(name='🔌 Connexion')

def update_connection_status(is_connected: bool):
    """Met à jour le statut de connexion"""
    connection_status.value = is_connected

class WebSocketClient:
    def __init__(self, user_id, host='localhost', port=8001, path='/v1/ws'):
        self.user_id = user_id
        self.host = host
        self.port = port
        self.path = path
        self.uri = f"ws://{host}:{port}{path}?user_id={urllib.parse.quote(user_id)}"
        self.websocket = None
        self.connected = False
        self._connect_task = None
        self._session_id = None
        self._lock = asyncio.Lock()
        print(f"WebSocket URI initialisée : {self.uri}")

    async def connect(self):
        if self.connected:
            return True
            
        if self._connect_task and not self._connect_task.done():
            return await self._connect_task
        
        self._connect_task = asyncio.create_task(self._connect())
        return await self._connect_task

    async def _connect(self):
        try:
            print(f"Tentative de connexion à : {self.uri}")
            print(f"Détails de la connexion - Host: {self.host}, Port: {self.port}, Path: {self.path}")
            
            # Ajouter du débogage sans utiliser extra_headers qui n'est pas supporté
            print(f"Informations de débogage - User-ID: {self.user_id}")
            
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=20,
                ping_timeout=20
            ) 
            self.connected = True
            update_connection_status(True)
            print("✅ Connexion WebSocket établie")
            
            # Ne pas attendre de message de bienvenue
            # Le serveur n'envoie pas de message de connexion
            self._session_id = None
            print("Pas de message de bienvenue attendu du serveur")
            
            # Lancer une tâche pour surveiller la connexion
            asyncio.create_task(self._monitor_connection())
            
            return True
        except Exception as e:
            print(f"❌ Erreur de connexion WebSocket : {type(e).__name__} - {str(e)}")
            self.connected = False
            update_connection_status(False)
            return False

    async def _monitor_connection(self):
        """Surveiller en continu l'état de la connexion"""
        try:
            while self.connected:
                try:
                    # Envoyer un ping
                    await self.websocket.ping()
                    await asyncio.sleep(15)  # Vérifier toutes les 15 secondes
                except websockets.exceptions.ConnectionClosed:
                    print("❌ Connexion WebSocket fermée de manière inattendue")
                    self.connected = False
                    update_connection_status(False)
                    await self.connect()
                    break
        except Exception as e:
            print(f"❌ Erreur lors de la surveillance de la connexion : {str(e)}")

    async def send_message(self, message):
        if not self.connected:
            print("🔗 Pas de connexion active, tentative de connexion...")
            success = await self.connect()
            if not success:
                print("❌ Échec de la connexion WebSocket")
                return {'status': 'error', 'message': 'Échec de la connexion'}

        try:
            async with self._lock:
                # Si message est déjà un dict, l'utiliser directement
                if isinstance(message, dict):
                    message_payload = message.copy()
                    # Ajouter user_id s'il n'est pas déjà présent
                    if 'user_id' not in message_payload:
                        message_payload["user_id"] = self.user_id
                    
                    # Adapter au nouveau format avec input_data si nécessaire
                    if 'input' in message_payload and 'input_data' not in message_payload:
                        message_payload["input_data"] = message_payload.pop("input")
                else:
                    # Format adapté pour LangGraph avec le nouveau champ input_data
                    message_payload = {
                        "input_data": message,  # Nouveau format avec input_data
                        "user_id": self.user_id
                    }
                
                # Ajouter l'ID de session si disponible
                if self._session_id:
                    message_payload["session_id"] = self._session_id
                
                # Convertir en JSON et envoyer
                json_payload = json.dumps(message_payload)
                print(f"📞 Envoi du payload au serveur: {json_payload}")
                print(f"🔍 Type de websocket: {type(self.websocket)}")
                
                try:
                    await self.websocket.send(json_payload)
                    print("💬 Message envoyé avec succès, en attente de réponse...")
                except Exception as send_error:
                    print(f"❌ Erreur lors de l'envoi du message au WebSocket: {type(send_error).__name__} - {str(send_error)}")
                    return {'status': 'error', 'message': f'Erreur lors de l\'envoi: {str(send_error)}'}
                
                # Attendre la réponse avec un timeout
                try:
                    print("⏳ Attente de réponse du serveur (timeout: 30s)...")
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                    print(f"📥 Réponse brute reçue: {response[:100]}...")
                    try:
                        response_data = json.loads(response)
                        print(f"📚 Réponse JSON parsée: {response_data}")
                    except json.JSONDecodeError as json_error:
                        print(f"❌ Erreur de décodage JSON: {str(json_error)}")
                        print(f"Réponse brute complète: {response}")
                        return {'status': 'error', 'message': 'Réponse non-JSON reçue'}
                    
                    # Ignorer les messages de ping
                    if isinstance(response_data, dict) and response_data.get('type') == 'ping':
                        print("🏓 Message ping reçu, en attente d'une réponse valide")
                        response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                        response_data = json.loads(response)
                    
                    return response_data
                except asyncio.TimeoutError:
                    return {'status': 'error', 'message': 'Timeout de la réponse'}
                except Exception as e:
                    return {'status': 'error', 'message': f'Erreur lors de la réception : {str(e)}'}
        
        except websockets.exceptions.ConnectionClosed:
            print("❌ Connexion perdue, tentative de reconnexion...")
            self.connected = False
            update_connection_status(False)
            success = await self.connect()
            if success:
                return await self.send_message(message)
            return {'status': 'error', 'message': 'Connexion perdue'}
        except websockets.exceptions.ConcurrencyError as e:
            print(f"❌ Erreur de concurrence WebSocket : {str(e)}")
            return {'status': 'error', 'message': 'Une requête est déjà en cours'}
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi du message : {type(e).__name__} - {str(e)}")
            self.connected = False
            update_connection_status(False)
            return {'status': 'error', 'message': f'Erreur: {str(e)}'}

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            update_connection_status(False)
            self._session_id = None

async def send_message_to_agent(message, context=None):
    """
    Envoi générique de message à l'agent LangGraph
    
    Args:
        message (str): Message à envoyer
        context (dict, optional): Contexte supplémentaire
    
    Returns:
        dict: Réponse de l'agent avec le contenu déjà parsé
    """
    try:
        print(f"\n📣 Début de send_message_to_agent avec message: {message}")
        # Format adapté pour LangGraph avec le nouveau format input_data
        payload = {
            "input_data": message  # Nouveau format avec input_data au lieu de input
        }
        
        # Ajouter le contexte si fourni
        if context:
            payload.update(context)
            
        print(f"📦 Payload préparé: {payload}")
        
        # Débug: Afficher le type de payload
        print("🔍 Type de payload:", type(payload))
        
        # Envoi du message - Envoyer directement le payload, pas en tant que string JSON
        print("💬 Envoi du message au WebSocket...")
        
        # Tentative d'envoi avec plusieurs essais
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🔄 Tentative d'envoi {attempt + 1}/{max_retries}")
                response = await websocket_client.send_message(payload)
                print(f"📥 Réponse reçue: {response}")
                break
            except Exception as e:
                print(f"⚠️ Erreur lors de l'envoi (tentative {attempt + 1}): {e}")
                if attempt == max_retries - 1:  # Dernière tentative
                    raise
                await asyncio.sleep(1)  # Attendre avant de réessayer
        
        # Vérifier si la réponse est valide
        if not response:
            print("⚠️ Réponse vide")
            return {'status': 'error', 'message': 'Réponse vide'}
            
        # Si la réponse contient déjà un statut d'erreur, le retourner directement
        if 'error' in response:
            print(f"⚠️ Réponse contient une erreur: {response['error']}")
            return {'status': 'error', 'message': response.get('error', 'Erreur inconnue')}
        
        if response:
            # Adaptation au format de réponse LangGraph
            if 'result' in response:
                # Nouveau format de réponse LangGraph (cohérent avec l'API REST)
                content = response.get('result', '')
                print(f"👍 Contenu extrait: {content[:100]}...")
                return {'status': 'success', 'data': {'response': content}}
            elif 'response' in response:
                # Ancien format pour compatibilité
                content = response.get('response', '')
                print(f"👍 Contenu extrait (ancien format): {content[:100]}...")
                return {'status': 'success', 'data': {'response': content}}
            else:
                # Autre format possible
                print(f"⚠️ Format de réponse non standard: {response}")
                return {'status': 'success', 'data': {'response': str(response)}}
        
        print(f"✅ Réponse de l'agent : {response}")
        return response
    
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du message : {e}")
        return {'status': 'error', 'message': f'Erreur lors de l\'envoi du message : {str(e)}'}




async def callback(contents: str, user: str, chat_instance: pn.chat.ChatInterface):
    global instance
    instance = chat_instance
    
    try:
        # Ne pas traiter les messages système ou assistant
        if user in ['Système', 'Assistant', '🤖 Assistant']:
            return

        # Envoyer le message à l'agent silencieusement
        print('callback send_message_to_agent ' , contents)
        
        # Message de débogage temporaire pour l'utilisateur
        chat_instance.send("Traitement de votre demande...", user='Système')
        
        # Envoyer le message et attendre la réponse
        print(f"📣 Envoi du message à l'agent: {contents}")
        response = await send_message_to_agent(contents)
        print(f"💯 Réponse finale reçue: {response}")
        
        # Effacer tous les messages
        chat_instance.objects = []
        
        # Traiter la réponse
        if response and response.get('status') == 'success':
            data = response.get('data', {})
            
            # Extraction du contenu principal
            content = data.get('response', '')
            
            # Envoi de la réponse de l'agent
            if content:
                # Nettoyer le contenu des caractères d'échappement
                content = content.replace("\\'", "'").replace('\\"', '"')
                
                # Envoyer la réponse textuelle
                print('callback send 2 chatbot as assistant message : ', content)
                chat_instance.send(content, user='🤖 Assistant')
            else:
                # Message d'erreur si pas de contenu
                chat_instance.send("Je n'ai pas pu générer de réponse. Veuillez réessayer.", user='Système')
            

        else:
            print("Erreur de réponse du serveur")
            
    except Exception as global_error:
        # Gestion des erreurs globales
        print(f"❌ Erreur globale : {global_error}")

# Variable globale pour stocker l'instance du chat
instance = None

# Initialisation du client WebSocket
websocket_client = WebSocketClient(
    user_id=user_id, 
    host=WEBSOCKET_CONFIG['host'], 
    port=WEBSOCKET_CONFIG['port'],
    path=WEBSOCKET_CONFIG['path']
)

# Initialisation de l'interface de chat
chat_interface = pn.chat.ChatInterface(
    callback=callback,
    callback_user="👤 Utilisateur",
    show_rerun=False,
    show_undo=False,
    show_clear=True,
    height=600,
    #sizing_mode='stretch_width',
    styles={
        '.chat-interface': {
            'width': '100%',
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'background': '#fff',
            'display': 'flex',
            'flex-direction': 'column'
        },
        '.chat-interface-messages': {
            'flex-grow': '1',
            'height': 'calc(100% - 60px)',  # Hauteur totale moins la zone de saisie
            'overflow-y': 'scroll',
            'font-size': '12px',
            'padding': '10px'
        },
        '.chat-interface-input': {
            'height': '40px',
            'font-size': '12px',
            'margin': '10px',
            'border-radius': '1px'
        },
        '.chat-interface-message': {
            'padding': '5px 10px',
            'margin': '5px 0',
            'border-radius': '1px'
        }
    }
)



# Mise en page
header = pn.Row(
    pn.pane.Markdown('# 💬 Chat Phidata'),
    pn.Spacer(width=20),
    connection_status,
    pn.Spacer(width=20),
    pn.Spacer(width=20),
    sizing_mode='stretch_width'
)

# Initialisation de Panel
pn.extension()



# Créer un conteneur dynamique pour la zone supérieure
dynamic_container = pn.Column(sizing_mode='stretch_width')


# Créer un layout avec une seule colonne
main_layout = pn.Column(
    chat_interface,  # Colonne unique pour le chat
    sizing_mode='stretch_width',
    styles={'background': '#ffffff'}
)

# Template avec le layout en une colonne
template = pn.template.FastListTemplate(
    title='Chat LangGraph',  # Mise à jour du titre
    header=header,
    main=[main_layout],  # Utiliser une liste avec le layout
    accent_base_color="#88d8b0",
    header_background="#88d8b0"
)

# Servir l'application
template.servable()