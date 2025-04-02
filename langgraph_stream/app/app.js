/**
 * Application JavaScript pour le chatbot LangGraph avec streaming WebSocket
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const startButton = document.getElementById('start-btn');
    
    // URL du WebSocket (à ajuster selon votre configuration)
    const wsUrl = 'ws://localhost:8000/ws/chat';
    let socket = null;
    
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
    function addStreamingChar(char) {
        // Vérification des marqueurs spéciaux
        if (char === '[END]') {
            console.log('Fin du flux de streaming');
            return;
        }
        
        if (char === '[START_MESSAGE]') {
            console.log('Début d\'un nouveau message');
            // Création d'un nouveau message bot
            addMessage('', 'bot');
            return;
        }
        
        if (char === '[END_MESSAGE]') {
            console.log('Fin du message actuel');
            const currentMessage = document.getElementById('current-bot-message');
            if (currentMessage) {
                currentMessage.id = ''; // Permettre un nouveau message
            }
            return;
        }
        
        // Gestion normale des caractères
        const currentMessage = document.getElementById('current-bot-message');
        if (currentMessage) {
            const charSpan = document.createElement('span');
            charSpan.classList.add('streaming-char');
            charSpan.textContent = char;
            currentMessage.appendChild(charSpan);
            
            // Scroll pour suivre le texte
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }
    }
    
    // Fonction pour se connecter au WebSocket
    function connectWebSocket() {
        // Fermer toute connexion existante
        if (socket) {
            socket.close();
        }
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log('Connexion WebSocket établie');
            
            // Envoi d'un message pour démarrer le workflow
            socket.send('start');
        };
        
        socket.onmessage = (event) => {
            // Réception des caractères en streaming
            const char = event.data;
            addStreamingChar(char);
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
    }
    
    // Gestion du clic sur le bouton de démarrage
    startButton.addEventListener('click', () => {
        // Ajout d'un message utilisateur
        addMessage('Démarrer la conversation', 'user');
        
        // Préparation d'un message bot vide pour le streaming
        addMessage('', 'bot');
        
        // Connexion au WebSocket
        connectWebSocket();
    });
});
