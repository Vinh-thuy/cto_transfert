# LangGraph Streaming avec WebSocket

Ce projet démontre l'utilisation de LangGraph avec un streaming WebSocket pour créer un chatbot simple.

## Structure du projet

```
langgraph_stream/
├── agent/               # Module de workflow LangGraph
│   ├── __init__.py
│   ├── nodes.py         # Nœuds du workflow (dire_bonjour, dire_bonsoir)
│   └── workflow.py      # Configuration du graphe LangGraph
├── api/                 # API WebSocket
│   ├── __init__.py
│   └── websocket_server.py
├── app/                 # Application frontend
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── main.py              # Point d'entrée principal
├── requirements.txt     # Dépendances Python
└── setup_env.sh         # Script de configuration de l'environnement virtuel avec UV
```

## Installation

### Avec UV (recommandé)

Ce projet utilise UV pour la gestion de l'environnement virtuel Python. UV est un gestionnaire de paquets et d'environnements virtuels rapide et moderne.

1. Configurez l'environnement virtuel avec UV :

```bash
./setup_env.sh
```

2. Activez l'environnement virtuel :

```bash
source .venv/bin/activate
```

### Alternative (sans UV)

Si vous préférez ne pas utiliser UV, vous pouvez installer les dépendances directement :

```bash
pip install -r requirements.txt
```

## Utilisation

1. Démarrez le serveur WebSocket :

```bash
python main.py
```

2. Ouvrez le fichier `app/index.html` dans votre navigateur ou servez les fichiers statiques avec un serveur HTTP simple.

3. Cliquez sur le bouton "Démarrer la conversation" pour voir le workflow LangGraph en action avec streaming.

## Fonctionnement

- Le workflow LangGraph contient deux nœuds : un qui dit "Bonjour" et un autre qui dit "Bonsoir"
- Les nœuds utilisent le streaming pour envoyer chaque caractère individuellement
- L'API WebSocket transmet ces caractères au frontend en temps réel
- Le frontend affiche les caractères avec une animation pour simuler une saisie en temps réel
