/**
 * Application JavaScript pour le chatbot LangGraph avec streaming WebSocket
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM chargé');
    const chatMessages = document.getElementById('chat-messages');
    const startButton = document.getElementById('start-btn');
    const userInput = document.getElementById('user-input');
    
    // URL du WebSocket (à ajuster selon votre configuration)
    const wsUrl = 'ws://localhost:8000/ws/chat';
    let socket = null;
    
    // Afficher un message de démarrage
    addMessage('Chatbot prêt. Posez votre question...', 'system');

    // Fonction pour ajouter un message au chat
    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message', type);
        
        if (type === 'bot') {
            // Pour les messages du bot, on prépare un conteneur pour le streaming
            messageDiv.id = 'current-bot-message';
        } else {
            messageDiv.textContent = content;
        }
        
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        
        return messageDiv;
    }
    
    // Fonction pour ajouter un caractère en streaming
    function addStreamingChar(char, agentId = 'default_agent') {
        const chatMessages = document.getElementById('chat-messages');
        
        // Recherche du conteneur de message spécifique à l'agent
        let currentMessage = document.getElementById(`current-${agentId}-message`);
        
        // Si aucun message n'existe pour cet agent, créer un nouveau conteneur
        if (!currentMessage) {
            // Conteneur principal pour l'agent
            const agentMessageContainer = document.createElement('div');
            agentMessageContainer.classList.add('agent-message-container');
            agentMessageContainer.dataset.agentId = agentId;
            
            // Conteneur de message
            currentMessage = document.createElement('div');
            currentMessage.id = `current-${agentId}-message`;
            currentMessage.classList.add('message', 'bot-message');
            
            // Étiquette de l'agent
            const agentLabel = document.createElement('div');
            agentLabel.classList.add('agent-label');
            agentLabel.textContent = `Agent: ${agentId}`;
            
            // Assemblage du conteneur
            agentMessageContainer.appendChild(agentLabel);
            agentMessageContainer.appendChild(currentMessage);
            
            // Ajout au conteneur principal de messages
            chatMessages.appendChild(agentMessageContainer);
        }
        
        // Ajout du caractère au message
        const charSpan = document.createElement('span');
        charSpan.textContent = char;
        currentMessage.appendChild(charSpan);
        
        // Défilement automatique
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    // Fonction pour effacer les messages précédents
    function clearPreviousMessages() {
        // Supprimer tous les messages existants
        const chatMessages = document.getElementById('chat-messages');
        chatMessages.innerHTML = '';
    }
    
    // Fonction pour se connecter au WebSocket
    function connectWebSocket(message = 'Bonjour') {
        // Fermer toute connexion existante
        if (socket) {
            socket.close();
        }
        
        // Effacer les messages précédents avant chaque nouvelle conversation
        clearPreviousMessages();
        
        try {
            console.log('Tentative de connexion à', wsUrl);
            socket = new WebSocket(wsUrl);
            
            socket.onopen = () => {
                console.log('Connexion WebSocket établie');
                console.log('Envoi du message:', message);
                
                // Envoyer le message
                socket.send(message);
                
                // Ajouter un message de confirmation
                addMessage('Message envoyé : ' + message, 'system');
            };
            
            socket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Utiliser l'agent_id si présent, sinon utiliser 'default'
                const agentId = data.agent_id || 'default_agent';
                
                if (data.type === 'stream') {
                    data.content.split('').forEach(char => {
                        addStreamingChar(char, agentId);
                    });
                }
                
                // Autres logiques de gestion des messages...
            };
            
            socket.onclose = (event) => {
                console.log('Connexion WebSocket fermée', event);
                
                if (event.wasClean) {
                    addMessage('Conversation terminée.', 'system');
                } else {
                    addMessage('Connexion perdue. Veuillez réessayer.', 'system');
                }
                
                // Renommer l'ID du message bot actuel pour permettre un nouveau message
                const currentBotMessage = document.getElementById('current-bot-message');
                if (currentBotMessage) {
                    currentBotMessage.id = '';
                }
            };
            
            socket.onerror = (error) => {
                console.error('Erreur WebSocket:', error);
                addMessage('Erreur de connexion. Veuillez réessayer.', 'system');
            };
        } catch (e) {
            console.error('Exception lors de la connexion WebSocket:', e);
            addMessage('Erreur de connexion: ' + e.message, 'system');
        }
    }
    
    // Fonction pour envoyer un message
    function sendMessage() {
        // Récupérer le message de l'utilisateur
        const message = userInput.value.trim();
        
        // Ne rien faire si le message est vide
        if (!message) {
            return;
        }
        
        console.log('Envoi du message:', message);
        
        // Ajouter le message utilisateur à l'historique
        addMessage(message, 'user');
        
        // Préparer un message bot vide pour le streaming
        addMessage('', 'bot');
        
        // Vider le champ de saisie
        userInput.value = '';
        
        // Donner le focus au champ de saisie
        userInput.focus();
        
        // Envoyer le message au serveur
        if (socket && socket.readyState === WebSocket.OPEN) {
            socket.send(message);
        } else {
            connectWebSocket(message);
        }
    }
    

    
    // Gestion du clic sur le bouton d'envoi
    startButton.addEventListener('click', sendMessage);
    
    // Gestion de la touche Entrée dans le champ de saisie
    userInput.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            event.preventDefault();
            sendMessage();
        }
    });
    
    // Connexion initiale au WebSocket lorsque l'utilisateur clique sur le bouton pour la première fois
    startButton.addEventListener('click', () => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            // Connexion initiale
            connectWebSocket('Bonjour');
        }
    }, { once: true });
});
