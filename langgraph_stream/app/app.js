/**
 * Application JavaScript pour le chatbot LangGraph avec streaming WebSocket
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const startButton = document.getElementById('start-btn');
    const userInput = document.getElementById('user-input');
    
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
        
        // Gestion des erreurs
        if (char.startsWith('Erreur :')) {
            console.error('Erreur détectée:', char);
            const currentMessage = document.getElementById('current-bot-message');
            if (currentMessage) {
                currentMessage.textContent = char;
                currentMessage.classList.add('error');
                currentMessage.id = ''; // Permettre un nouveau message
            } else {
                // Créer un nouveau message d'erreur si nécessaire
                const errorMessage = addMessage(char, 'bot error');
                errorMessage.id = '';
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
    function connectWebSocket(message = 'Bonjour') {
        // Fermer toute connexion existante
        if (socket) {
            socket.close();
        }
        
        socket = new WebSocket(wsUrl);
        
        socket.onopen = () => {
            console.log('Connexion WebSocket établie');
            console.log('Envoi du message:', message);
            
            // Envoyer le message
            socket.send(message);
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
