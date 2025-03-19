from langchain_openai import ChatOpenAI
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool

from dotenv import load_dotenv
import os
import logging
import random

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Créer un logger spécifique pour le suivi des outils
tool_logger = logging.getLogger("tool_usage")
tool_logger.setLevel(logging.DEBUG)

from pydantic import BaseModel, Field


# Charger les variables d'environnement
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", openai_api_key)


llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)

# Définition d'un outil pour générer des titres accrocheurs
@tool
def generate_catchy_title(theme: str) -> str:
    """Génère un titre accrocheur pour une histoire basée sur un thème donné.
    
    Args:
        theme: Le thème principal de l'histoire
        
    Returns:
        Un titre accrocheur pour l'histoire
    """
    tool_logger.debug(f"[TOOL] generate_catchy_title appelé avec le thème: '{theme}'")
    
    # Liste de modèles de titres accrocheurs
    title_templates = [
        "Le Secret de {}",
        "Quand {} Rencontre le Destin",
        "Les Mystères de {}",
        "{} : Une Histoire Inattendue",
        "Le Dernier {} du Monde",
        "L'Incroyable Aventure de {}",
        "{} : La Vérité Cachée",
        "Le Jour où {} a Changé"
    ]
    
    # Sélection aléatoire d'un modèle et formatage avec le thème
    selected_template = random.choice(title_templates)
    title = selected_template.format(theme.capitalize())
    
    tool_logger.info(f"[TOOL] Titre généré: '{title}' (modèle: '{selected_template}')")
    return title

# Lier l'outil au LLM
llm_with_tools = llm.bind_tools([generate_catchy_title])



# Schema for structured output to use as routing logic
class Route(BaseModel):
    step: Literal["chatbot", "story", "joke"] = Field(
        None, description="The next step in the routing process is to direct the request to 'story', 'joke', or 'chatbot' based on the user's request."
    )


# Augment the LLM with schema for structured output
router = llm.with_structured_output(Route)


# State
class State(TypedDict):
    input: str
    decision: str
    output: str
    user_approval: bool


# Nodes
def user_interrupt(state: State):
    """Demande la validation de l'utilisateur avant d'exécuter le nœud llm_call_1"""
    logging.info("=== Demande de validation utilisateur avant d'exécuter le nœud story (llm_call_1) ===")
    logging.info(f"Entrée reçue: '{state['input']}'")
    
    # Extraire le thème principal de l'entrée
    theme = state['input'].split()[-1] if len(state['input'].split()) > 0 else "mystère"
    
    # Message pour l'utilisateur avec instructions claires
    message = f"VALIDATION_REQUISE: Je vais générer une histoire sur le thème '{theme}'. Souhaitez-vous continuer avec cette action ? Répondez par 'oui' pour continuer ou 'non' pour annuler."
    
    # Dans un système réel, nous ne définissons pas user_approval ici
    # Il sera fourni par l'utilisateur lors d'une interaction ultérieure
    
    logging.info("En attente de la validation utilisateur...")
    logging.info("=== Interruption pour validation utilisateur ===")
    
    # Retourner l'output avec un message clair indiquant qu'une validation est requise
    # Le client devra détecter ce préfixe pour savoir qu'une validation est nécessaire
    return {"output": message, "requires_validation": True}


def llm_call_1(state: State):
    """Write a story with a catchy title"""
    logging.info("=== Début du traitement du nœud story (llm_call_1) ===")
    logging.info(f"Entrée reçue: '{state['input']}'")
    
    # Extraire le thème principal de l'entrée
    theme = state['input'].split()[-1] if len(state['input'].split()) > 0 else "mystère"
    logging.info(f"Thème extrait pour le titre: '{theme}'")
    
    # Utiliser l'outil pour générer un titre accrocheur
    logging.info("Appel de l'outil generate_catchy_title...")
    title = generate_catchy_title(theme)
    logging.info(f"Titre généré avec succès: '{title}'")
    
    # Créer le prompt avec le titre généré
    system_prompt = "Vous êtes un écrivain créatif. Écrivez une histoire avec de l'humour noir en quelques lignes (2 max) avec le titre suivant et basée sur :"
    full_prompt = f"{system_prompt} '{title}' - Thème: {state['input']}"
    logging.debug(f"Prompt complet envoyé au LLM: '{full_prompt}'")
    
    # Utiliser le LLM avec l'outil lié
    logging.info("Appel du LLM avec les outils liés...")
    result = llm_with_tools.invoke(full_prompt)
    logging.debug(f"Réponse brute du LLM: '{result.content}'")
    
    # Vérifier si le LLM a utilisé des outils dans sa réponse
    if hasattr(result, 'tool_calls') and result.tool_calls:
        logging.info(f"Le LLM a utilisé {len(result.tool_calls)} outil(s) dans sa réponse")
        for i, tool_call in enumerate(result.tool_calls):
            logging.info(f"Outil {i+1}: {tool_call['name']} avec arguments: {tool_call['args']}")
    else:
        logging.info("Le LLM n'a pas utilisé d'outils supplémentaires dans sa réponse")
    
    # Formater la sortie avec le titre
    formatted_output = f"**{title}**\n\n{result.content}"
    logging.info(f"Sortie formatée avec titre: '{formatted_output[:50]}...'")
    logging.info("=== Fin du traitement du nœud story (llm_call_1) ===")
    
    return {"output": formatted_output}


def llm_call_2(state: State):
    """Write a joke"""
    logging.info("Calling llm_call_2 for joke generation.")
    result = llm.invoke(state["input"])
    return {"output": result.content}


def llm_call_3(state: State):
    """Write a story"""
    logging.info("Calling llm_call_3 for Chatbot info.")
    system_prompt = "Vous êtes un chatbot LangGraph. Votre rôle est d'interagir avec les utilisateurs en répondant à leurs questions et en fournissant des informations utiles. Vous pouvez générer des histoires, des poèmes ou des blagues en fonction des demandes des utilisateurs. Répondez de manière claire et concise."
    full_prompt = f"{system_prompt} {state['input']}"
    result = llm.invoke(full_prompt)
    return {"output": result.content}


router_prompt = """
You are a senior specialist responsible for classifying incoming requests and questions. Depending on the nature of the question, you must return a decision that will be used to route the question to the appropriate team or agent.

Here are the three possibilities to consider:

1. If the question is related to stories, return "story".
2. If the question is related to jokes, return "joke".
3. If the question concerns information, question ou use case for the chatbot, return "chatbot".

Your response must be a single word: story, joke, or chatbot.
"""

def llm_call_router(state: State):
    """Route the input to the appropriate node"""
    # Vérifier si l'entrée est une simple réponse 'oui'/'non' à une demande de validation
    input_text = state.get("input", "").lower().strip()
    if input_text in ['oui', 'yes', 'y', 'ok', 'continue', 'continuer', 'non', 'no', 'n', 'cancel', 'annuler']:
        logging.info(f"=== Détection d'une simple réponse de validation: '{input_text}' ===")
        
        # Pour une réponse positive, forcer la décision à 'story' sans appeler le LLM
        if input_text in ['oui', 'yes', 'y', 'ok', 'continue', 'continuer']:
            logging.info("=== Réponse POSITIVE détectée, forçage de la décision à 'story' ===")
            return {"decision": "story", "user_approval": True, "is_validation_response": True}
        else:
            # Pour une réponse négative, on peut garder la logique habituelle
            logging.info("=== Réponse NÉGATIVE détectée ===")
    
    # Vérifier si nous devons contourner le routeur (pour les réponses de validation)
    if state.get("bypass_router", False):
        logging.info("=== Contournement du routage LLM détecté ===")
        logging.info(f"Conservation de la décision existante: '{state.get('decision', 'story')}'")
        # Retourner l'état tel quel, sans appeler le LLM
        return state
    
    # Ancienne vérification pour la rétrocompatibilité
    if state.get("is_validation_response", False):
        logging.info("=== Contournement du routage pour une réponse de validation (ancien flag) ===")
        logging.info(f"Conservation de la décision existante: '{state.get('decision', 'story')}'")
        return {"decision": state.get("decision", "story")}
    
    # Si l'entrée est vide, conserver la décision existante (pour les réponses de validation)
    if not state.get("input", "").strip():
        logging.info("=== Entrée vide détectée, contournement du routage ===")
        logging.info(f"Conservation de la décision existante: '{state.get('decision', 'story')}'")
        return state
    
    # Sinon, procéder au routage normal
    logging.info("=== Début du routage de l'entrée ===")
    logging.info(f"Entrée à router: '{state['input']}'")
    decision = router.invoke(
        [
            SystemMessage(
                content=router_prompt
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    logging.info(f"Décision de routage: '{decision.step}'")
    
    if decision.step == "story":
        logging.info("La requête sera traitée par le nœud story (llm_call_1) qui utilise l'outil generate_catchy_title")
    else:
        logging.info(f"La requête sera traitée par le nœud {decision.step}")
    
    logging.info("=== Fin du routage de l'entrée ===")
    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Vérifier si c'est une réponse de validation
    if state.get("is_validation_response", False) or state.get("bypass_router", False):
        logging.info("=== Réponse de validation détectée dans route_decision ===")
        logging.info(f"Décision: {state.get('decision', 'non spécifiée')}, Approbation: {state.get('user_approval', 'non spécifiée')}")
        
        # Si l'utilisateur a approuvé et que la décision est 'story', aller directement à llm_call_1
        if state.get("user_approval", False) is True and state.get("decision", "") == "story":
            logging.info("Validation POSITIVE pour 'story', routage direct vers llm_call_1")
            return "llm_call_1"
        # Si l'utilisateur a approuvé et que la décision est 'chatbot', aller directement à llm_call_3
        elif state.get("user_approval", False) is True and state.get("decision", "") == "chatbot":
            logging.info("Validation POSITIVE pour 'chatbot', routage direct vers llm_call_3")
            return "llm_call_3"
        # Si l'utilisateur n'a pas approuvé, terminer le flux
        elif state.get("user_approval", False) is False:
            logging.info("Validation NEGATIVE, fin du flux")
            return END
    
    # Si l'utilisateur a déjà donné son approbation et que nous avons une décision 'story',
    # aller directement au noeud llm_call_1 sans passer par l'interruption
    if "user_approval" in state and state["user_approval"] is True and state["decision"] == "story":
        logging.info("Approbation utilisateur déjà reçue, routage direct vers le noeud story (llm_call_1)")
        return "llm_call_1"  # Aller directement au noeud story sans interruption
    
    # Sinon, suivre le flux normal
    if state["decision"] == "story":
        logging.info("Routage vers le nœud d'interruption utilisateur")
        return "user_interrupt"  # Rediriger vers l'interruption utilisateur avant llm_call_1
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "chatbot":
        return "llm_call_3"


# Build workflow
logging.info("Construction du graphe d'état...")
router_builder = StateGraph(State)

# Add nodes
logging.info("Ajout des nœuds au graphe...")
router_builder.add_node("llm_call_router", llm_call_router)
router_builder.add_node("user_interrupt", user_interrupt)  # Nœud pour l'interruption utilisateur
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)

# Add edges to connect nodes
logging.info("Ajout des arêtes au graphe...")
# Arête de départ vers le routeur
router_builder.add_edge(START, "llm_call_router")

# Arêtes conditionnelles depuis le routeur
logging.info("Ajout des arêtes conditionnelles depuis le routeur...")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "user_interrupt": "user_interrupt",  # Redirection vers le nœud d'interruption
        "llm_call_1": "llm_call_1",  # Redirection directe vers le nœud story si approbation déjà donnée
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)

# Fonction conditionnelle pour décider si on continue après l'interruption
def check_user_approval(state: State):
    # Vérifier si la décision est spécifiée dans l'état
    decision = state.get("decision", None)
    
    # Débogage: Afficher l'état complet pour comprendre ce qui est reçu
    logging.info(f"check_user_approval - État reçu: {state}")
    
    # Si user_approval est présent et True, continuer vers le noeud approprié en fonction de la décision
    if "user_approval" in state and state["user_approval"] is True:
        logging.info(f"Validation utilisateur reçue (POSITIVE): Continuer vers le noeud approprié (décision: {decision})")
        
        # IMPORTANT: Forcer le routage vers llm_call_1 si la décision est 'story'
        # indépendamment des autres facteurs
        if decision == "story":
            logging.info("Routage FORCÉ vers le noeud story (llm_call_1)")
            return "llm_call_1"  # Noeud pour générer une histoire
        elif decision == "chatbot":
            logging.info("Routage vers le noeud chatbot (llm_call_3)")
            return "llm_call_3"  # Noeud pour le chatbot
        else:
            # Par défaut, utiliser llm_call_1 si aucune décision spécifique n'est fournie
            logging.info("Aucune décision spécifique, routage par défaut vers llm_call_1")
            return "llm_call_1"
    
    # Si user_approval est présent et False, terminer le flux
    elif "user_approval" in state and state["user_approval"] is False:
        logging.info("Validation utilisateur reçue: Terminer le flux (refus)")
        return END  # Terminer le flux si l'utilisateur n'approuve pas
    
    # Si user_approval n'est pas présent, terminer le flux avec un message d'attente
    else:
        logging.info("Validation utilisateur non reçue: Terminer avec message d'attente")
        # Au lieu de boucler, nous terminons le flux et laissons le client gérer l'attente
        return END

# Ajouter les arêtes conditionnelles pour le nœud d'interruption
logging.info("Ajout des arêtes conditionnelles pour le nœud d'interruption...")
router_builder.add_conditional_edges(
    "user_interrupt",  # Noeud source
    check_user_approval,  # Fonction de décision
    {
        # Mapping des valeurs de retour de check_user_approval vers les noeuds cibles
        "llm_call_1": "llm_call_1",  # Si l'utilisateur approuve
        "llm_call_3": "llm_call_3",  # Si l'utilisateur approuve pour le chatbot
        END: END,  # Si l'utilisateur n'approuve pas ou en attente
    }
)

# Ajouter les arêtes finales pour terminer les flux
logging.info("Ajout des arêtes finales...")
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile workflow
logging.info("Compilation du workflow...")
try:
    router_workflow = router_builder.compile()
    logging.info("Workflow compilé avec succès!")
except Exception as e:
    logging.error(f"Erreur lors de la compilation du workflow: {e}")
    raise


# Invoke
# Aucun test automatique n'est nécessaire ici
# L'agent sera testé via l'API et l'interface utilisateur
