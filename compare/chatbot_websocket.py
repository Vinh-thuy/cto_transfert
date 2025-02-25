import param
import panel as pn
import asyncio
import json
import websockets
import os
import threading
import urllib.parse
from io import StringIO
import html
import hvplot.pandas
import numpy as np
import pandas as pd

# Configuration WebSocket
WEBSOCKET_CONFIG = {
    'host': os.getenv('WEBSOCKET_HOST', 'localhost'),  
    'port': int(os.getenv('WEBSOCKET_PORT', 8000)),       
    'path': '/v1/ws'
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
        from bokeh.io import curdoc
        
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
            self.websocket = await websockets.connect(
                self.uri,
                ping_interval=20,
                ping_timeout=20
            ) 
            self.connected = True
            update_connection_status(True)
            print("✅ Connexion WebSocket établie")
            
            # Recevoir et traiter le message de connexion
            connection_response = await self.websocket.recv()
            connection_data = json.loads(connection_response)
            
            if connection_data.get('status') == 'success':
                self._session_id = connection_data.get('data', {}).get('session_id')
                print(f"📡 Session ID obtenu : {self._session_id}")
            
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

    async def send_message(self, message: str):
        if not self.connected:
            success = await self.connect()
            if not success:
                return {'status': 'error', 'message': 'Échec de la connexion'}

        try:
            async with self._lock:
                # Préparer le message avec les informations de session
                message_payload = {
                    "query": message,
                    "user_id": self.user_id,
                    "model_id": model_id
                }
                
                # Ajouter l'ID de session si disponible
                if self._session_id:
                    message_payload["session_id"] = self._session_id
                
                await self.websocket.send(json.dumps(message_payload))
                
                # Attendre la réponse avec un timeout
                try:
                    response = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                    response_data = json.loads(response)
                    
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
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi du message : {type(e).__name__} - {str(e)}")
            self.connected = False
            update_connection_status(False)
            return {'status': 'error', 'message': f'Erreur: {str(e)}'}
        except websockets.exceptions.ConcurrencyError as e:
            print(f"❌ Erreur de concurrence WebSocket : {str(e)}")
            return {'status': 'error', 'message': 'Une requête est déjà en cours'}

    async def close(self):
        if self.websocket:
            await self.websocket.close()
            self.connected = False
            update_connection_status(False)
            self._session_id = None

async def send_message_to_agent(message, context=None):
    """
    Envoi générique de message à l'agent
    
    Args:
        message (str): Message à envoyer
        context (dict, optional): Contexte supplémentaire
    
    Returns:
        dict: Réponse de l'agent avec le contenu déjà parsé
    """
    try:
        # Préparer le payload
        payload = {
            "query": message
        }
        
        # Ajouter le contexte si fourni
        if context:
            payload["context"] = context
        
        # Envoi du message
        response = await websocket_client.send_message(json.dumps(payload))
        
        if response and response.get('status') == 'success':
            data = response.get('data', {})
            content = data.get('response', '')
            
            try:
                # Parser le contenu JSON une seule fois
                content_json = json.loads(content)
                # Remplacer la réponse brute par le contenu parsé
                data['response'] = content_json.get('content', '')
                response['data'] = data
            except json.JSONDecodeError:
                # Garder le contenu tel quel si ce n'est pas du JSON
                pass
        
        print(f"✅ Réponse de l'agent : {response}")
        return response
    
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du message : {e}")
        return {'status': 'error', 'message': f'Erreur lors de l\'envoi du message : {str(e)}'}

def create_dynamic_widget(widget_config):
    """
    Créer un widget dynamique basé sur sa configuration
    
    Args:
        widget_config (dict): Configuration du widget
    
    Returns:
        Un widget Panel
    """
    widget_type = widget_config.get('type', '').lower()
    print(f"create_dynamic_widget - Type de widget : {widget_type}")
    
    if widget_type == 'select':
        # Configuration du widget Select
        options = widget_config.get('options', [])
        print(f"create_dynamic_widget - Options : {options}")
        name = widget_config.get('name', 'Sélectionnez une option')
        print(f"create_dynamic_widget - Nom : {name}")
        
        # Créer le widget Select
        select_widget = pn.widgets.Select(
            name=name, 
            options=options,
            value=None
        )
        print('create_dynamic_widget - button_widget')
        
        # Créer un bouton de validation spécifique pour ce widget
        button_widget = pn.widgets.Button(
            name='Valider',
            button_type='primary'
        )

        # Callback pour le bouton
        async def on_button_click(event):
            current_value = select_widget.value
            print(f"Valeur sélectionnée lors du clic : {current_value}")
            if current_value:
                instance.send(f"Option sélectionnée : {current_value}", user='👤 Utilisateur')
                response = await send_message_to_agent(
                    current_value,
                    context={
                        "widget_type": "select",
                        "widget_name": select_widget.name
                    }
                )
        print('create_dynamic_widget - button_widget on_click')
        button_widget.on_click(on_button_click)
        
        # Retourner les deux widgets
        return pn.Row(select_widget, button_widget)
    
    # elif widget_type == 'button':
    #     # Configuration du widget Button
    #     label = widget_config.get('label', 'Valider')
    #     options = widget_config.get('options', [])
        
    #     # Créer le widget Button
    #     button_widget = pn.widgets.Button(
    #         name=label, 
    #         button_type='primary'
    #     )
        
    #     # Ajouter un attribut personnalisé pour stocker les options
    #     button_widget.options = options
    #     button_widget.label = label
        
    #     return button_widget
    
    else:
        raise ValueError(f"Type de widget non supporté : {widget_type}")

def create_widget_callback(select_widget, button_widget, options):
    """
    Crée un callback pour valider la sélection
    
    Args:
        select_widget (pn.widgets.Select): Widget de sélection
        button_widget (pn.widgets.Button): Bouton de validation
        options (list): Liste des options disponibles
    """
    def validate_and_send(event=None):
        """Callback pour valider et envoyer la sélection"""
        try:
            # Récupérer la valeur sélectionnée
            selected_value = select_widget.value
            
            # Vérifier si une option a été sélectionnée
            if not selected_value:
                instance.send("Veuillez sélectionner une option valide.", user='⚠️ Erreur')
                return
                
            if selected_value:
                print(f"📤 Option sélectionnée : {selected_value}")
                
                # Fonction asynchrone pour envoyer le message
                async def process_widget_message():
                    # Envoyer d'abord le message de l'utilisateur
                    print('validate_and_send send to chatbot as user message selected_value: ', selected_value)
                    instance.send(selected_value, user='👤 Utilisateur')
                    
                    # Envoyer le message à l'agent
                    print('validate_and_send send to agent message : selected_value et select_widget.name', selected_value, select_widget.name)
                    response = await send_message_to_agent(
                        selected_value, 
                        context={
                            "widget_type": "select",
                            "widget_name": select_widget.name
                        }
                    )
                    
                    # Traiter la réponse de l'agent
                    if response and response.get('status') == 'success':
                        # Extraire et afficher le contenu de la réponse
                        content = response.get('data', {}).get('response', '')
                        if content:
                            print('validate_and_send send to chatbot as assistant message content: ', content)
                            instance.send(content, user='🤖 Assistant')
                        
                        # Ajouter les widgets potentiels
                        widgets = response.get('data', {}).get('widgets', [])
                        if widgets:
                            dynamic_widgets = [
                                create_dynamic_widget(widget_config) 
                                for widget_config in widgets
                            ]
                            
                            # Message d'introduction pour les widgets
                            print('validate_and_send send to chatbot as system Message introduction pour les widgets')
                            instance.send("Informations complémentaires :", user='💡 Système')
                            
                            # Créer un layout horizontal avec tous les widgets
                            widget_layout = pn.Row(*dynamic_widgets)
                            instance.append(widget_layout)
                
                # Lancer la tâche asynchrone
                asyncio.create_task(process_widget_message())
                
                # Supprimer les widgets après l'envoi
                instance.objects = [
                    obj for obj in instance.objects 
                    if not (isinstance(obj, pn.Row) and 
                            any(isinstance(w, (pn.widgets.Select, pn.widgets.Button)) for w in obj.objects))
                ]
            else:
                instance.send("Veuillez sélectionner une option avant de valider.", user='⚠️ Système')
            
        except Exception as e:
            print(f"❌ Erreur dans le callback : {e}")
    
    return validate_and_send

def create_widget_layout(widget_config):
    """
    Crée une disposition de widgets basée sur la configuration
    
    Args:
        widget_config (dict): Configuration des widgets
    
    Returns:
        pn.Row: Layout avec les widgets configurés
    """
    print('create_widget_layout - Entree')
    def get_widget_value(widget):
        """Récupère la valeur d'un widget selon son type"""
        print(f"get_widget_value - Type de widget : {type(widget)}")
        
        if isinstance(widget, pn.layout.Row):
            # Pour un Row, récupérer la valeur du Select s'il existe
            for child in widget:
                if isinstance(child, pn.widgets.Select):
                    value = child.value
                    print(f"get_widget_value - Select widget value : {value}, options: {child.options}")
                    return value
            return None
            
        if isinstance(widget, pn.widgets.Select):
            value = widget.value
            print(f"get_widget_value - Select widget value : {value}, options: {widget.options}")
            return value
            
        if isinstance(widget, (pn.widgets.TextInput, pn.widgets.TextAreaInput)):
            return widget.value

        elif isinstance(widget, pn.widgets.Checkbox):
            return widget.value
        elif isinstance(widget, pn.widgets.DatePicker):
            return widget.value.strftime('%Y-%m-%d') if widget.value else None
        elif isinstance(widget, pn.widgets.Button):
            return widget.name
        return None

    def create_widget_context(widget):
        """Crée le contexte pour un widget"""
        widget_type = type(widget).__name__.lower()
        return {
            "widget_type": widget_type,
            "widget_name": getattr(widget, 'name', widget_type),
            "widget_value": get_widget_value(widget)
        }

    # Identifier les types de widgets disponibles
    widgets = []
    
    #widget = create_dynamic_widget(widget_config)
    widget_type = widget_config.get('type', '').lower()

    print('widget_type : ', widget_type)
    if widget_type == 'select':
        print('ICI 02')
        # Configuration du widget Select
        options = widget_config.get('options', [])
        print(f"create_dynamic_widget - Options : {options}")
        name = widget_config.get('name', 'Sélectionnez une option')
        print(f"create_dynamic_widget - Nom : {name}")
        
        # Créer le widget Select
        print('ICI 03')
        select_widget = pn.widgets.Select(
            name=name, 
            options=options,
            value=None
        )
        print('create_dynamic_widget - button_widget')
        
        # Créer un bouton de validation spécifique pour ce widget
        button_widget = pn.widgets.Button(
            name='Valider',
            button_type='primary'
        )

        # Callback pour le bouton
        def on_button_click(event):
            current_value = select_widget.value
            print(f"Valeur sélectionnée lors du clic : {current_value}")
            if current_value:
                instance.send(f"Option sélectionnée : {current_value}", user='👤 Utilisateur')
                response = send_message_to_agent(
                    current_value,
                    context={
                        "widget_type": "select",
                        "widget_name": select_widget.name
                    }
                )
        print('create_dynamic_widget - button_widget on_click')
        button_widget.on_click(on_button_click)

        return pn.Row(select_widget, button_widget)



    #print('create_widget_layout - widget : ', widget)
    # if widget:
    #     widgets.append(widget)

    # Ajouter un bouton de validation s'il n'y en a pas déjà
    print(' ICI 1 : ')
    # if not any(isinstance(w, pn.widgets.Button) for w in widgets):
    #     print(' if not any(isinstance(w, pn.widgets.Button) for w in widgets): 2')
    #     button_widget = create_dynamic_widget({
    #         'type': 'button',
    #         'label': 'Valider'
    #     })
    #     widgets.append(button_widget)
    
    # Créer un callback générique pour tous les widgets
    async def generic_widget_callback(event):
        try:
            # Récupérer le widget qui a déclenché l'action
            trigger_widget = event.obj if hasattr(event, 'obj') else None
            if not trigger_widget:
                return

            # Récupérer les valeurs de tous les widgets
            widget_values = {}
            widget_contexts = []
            for widget in widgets:
                value = get_widget_value(widget)
                print(f"generic_widget_callback collecte des value de Widget: {widget.name}, Valeur: {value}")

                # Ignorer les valeurs vides pour les widgets Select
                if isinstance(widget, pn.widgets.Select) and (value is None or value == ""):
                    print(f"Ignorer la valeur vide du widget Select: {widget.name}")
                    continue

                if value is not None:
                    widget_values[widget.name] = value
                    widget_contexts.append(create_widget_context(widget))

            # Vérifier si nous avons des valeurs à envoyer
            if not widget_values:
                instance.send("Veuillez remplir au moins un champ", user='⚠️ Système')
                return

            # Envoyer les valeurs sélectionnées
            value_message = ", ".join([f"{k}: {v}" for k, v in widget_values.items()])
            print('generic_widget_callback send to chatbot as user message value_message: ', value_message)
            instance.send(f"Valeurs sélectionnées : {value_message}", user='👤 Utilisateur')
            
            # Supprimer les widgets actuels
            instance.objects = [
                obj for obj in instance.objects 
                if not (isinstance(obj, pn.Row) and 
                        any(isinstance(w, tuple(set(type(widget) for widget in widgets))) for w in obj.objects))
            ]
            
            # Envoyer à l'agent
            print('generic_widget_callback send to agent json.dumps(widget_values) et widget_contexts', json.dumps(widget_values), widget_contexts)
            response = await send_message_to_agent(
                json.dumps(widget_values),
                context={
                    "widgets": widget_contexts
                }
            )
            
            # Traiter la réponse
            if response and response.get('status') == 'success':
                content = response.get('data', {}).get('response', '')
                if content:
                    print('generic_widget_callback send to chatbot as assistant content: ', content)
                    instance.send(content, user='🤖 Assistant')

        except Exception as e:
            print(f"❌ Erreur dans le callback widget : {e}")
            instance.send(f"Une erreur est survenue : {str(e)}", user='⚠️ Système')
    
    # Ajouter le callback générique à tous les widgets qui supportent les événements
    print(' ICI 2 : ')
    # for widget in widgets:
    #     if hasattr(widget, 'on_click'):
    #         widget.on_click(generic_widget_callback)
    #     elif hasattr(widget, 'param.watch'):
    #         widget.param.watch(generic_widget_callback, 'value')
    
    # Créer un layout horizontal avec tous les widgets
    print(' ICI 3 : ')
    #return pn.Row(*widgets) if widgets else None
    return pn.Column(select_widget, button_widget)

async def callback(contents: str, user: str, chat_instance: pn.chat.ChatInterface):
    global instance
    instance = chat_instance
    
    try:
        # Ne pas traiter les messages système ou assistant
        if user in ['Système', 'Assistant', '🤖 Assistant']:
            return

        # Envoyer le message à l'agent silencieusement
        print('callback send_message_to_agent ' , contents)
        response = await send_message_to_agent(contents)
        
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
            
            # Gestion des widgets dynamiques
            widgets = data.get('widgets', [])
            print('callback widgets: ', widgets)
            if widgets:
                print("Widgets reçus de l'agent")
                # Créer un seul widget layout pour tous les widgets
                select_config = next((w for w in widgets if w.get('type') == 'select'), None)
                print('callback select_config: ', select_config)
                #button_config = next((w for w in widgets if w.get('type') == 'button'), None)
                
                if select_config:
                    print("Configuration du widget select trouvée")
                    widget_row = create_widget_layout(select_config)
                    # Assurez-vous que widget_row est ajouté à la mise en page principale
                    chat_instance.send(widget_row, user='🤖 Assistant')
            else:
                # Remettre l'iframe si pas de widgets
                dynamic_container.clear()
                dynamic_container.append(iframe_pane)
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

# Pas de message d'accueil pour garder l'interface propre

# Switch d'agent
agent_options = ['agent_base']
agent_switch = pn.widgets.Select(
    name='🔄 Agent actif',
    options=agent_options,
    value='user_proxy',
    width=150
)

async def on_agent_switch(event):
    """Gère le changement d'agent"""
    new_agent = event.new
    try:
        switch_result = await websocket_client.send_message(f"switch_agent {new_agent}")
        if not (switch_result and switch_result.get('status') == 'success'):
            print(f"Erreur lors du changement d'agent : {switch_result}")
    except Exception as e:
        print(f"Erreur lors du changement d'agent : {str(e)}")

agent_switch.param.watch(on_agent_switch, 'value')

# Mise en page
header = pn.Row(
    pn.pane.Markdown('# 💬 Chat Phidata'),
    pn.Spacer(width=20),
    connection_status,
    pn.Spacer(width=20),
    agent_switch,
    pn.Spacer(width=20),
    sizing_mode='stretch_width'
)

# Initialisation de Panel
pn.extension()

def create_plot_iframe():
    """Crée un iframe avec un graphique"""
    # Set seed for reproducibility
    np.random.seed(1)

    # Create a time-series data frame
    idx = pd.date_range("1/1/2000", periods=1000)
    df = pd.DataFrame(np.random.randn(1000, 4), index=idx, columns=list("ABCD")).cumsum()

    # Plot the data using hvplot
    plot = df.hvplot()

    # Save the plot
    plot_file = StringIO()
    hvplot.save(plot, plot_file)
    plot_file.seek(0)

    # Read the HTML content and escape it
    html_content = plot_file.read()
    escaped_html = html.escape(html_content)

    # Create and return iframe HTML
    return f'<iframe srcdoc="{escaped_html}" style="width:100%; height:350px;" frameborder="0"></iframe>'

# Créer un conteneur dynamique pour la zone supérieure
dynamic_container = pn.Column(sizing_mode='stretch_width')

# Créer l'iframe
iframe_html = create_plot_iframe()
iframe_pane = pn.pane.HTML(iframe_html, height=600, sizing_mode="stretch_width")
dynamic_container.append(iframe_pane)

# Créer un layout en deux colonnes
main_layout = pn.Row(
    pn.Column(dynamic_container, width=700),  # Colonne de gauche pour iframe/widgets
    pn.Column(chat_interface, width=700),     # Colonne de droite pour le chat
    sizing_mode='stretch_width',
    styles={'background': '#ffffff'}
)

# Template avec le layout en deux colonnes
template = pn.template.FastListTemplate(
    title='Chat Phidata',
    header=header,
    main=[main_layout],  # Utiliser une liste avec le layout
    accent_base_color="#88d8b0",
    header_background="#88d8b0"
)

# Servir l'application
template.servable()