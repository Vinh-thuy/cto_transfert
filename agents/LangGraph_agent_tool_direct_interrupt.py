from langchain_openai import ChatOpenAI
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint import Checkpoint
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
openai_api_key = "sk-proj-zrA4zBQ285cK3j"
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
    
    # Message pour l'utilisateur
    message = f"Je vais générer une histoire sur le thème '{theme}'. Souhaitez-vous continuer avec cette action ?"
    
    # Simuler une réponse utilisateur (dans un vrai système, cela serait interactif)
    # Pour les besoins de la démonstration, nous supposons que l'utilisateur approuve
    user_approval = True  # Dans un système réel, cela viendrait d'une entrée utilisateur
    
    logging.info(f"Validation utilisateur: {user_approval}")
    logging.info("=== Fin de la demande de validation utilisateur ===")
    
    return {"user_approval": user_approval, "output": message}


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
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "user_interrupt"  # Rediriger vers l'interruption utilisateur avant llm_call_1
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "chatbot":
        return "llm_call_3"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
router_builder.add_node("user_interrupt", user_interrupt)  # Nouveau nœud pour l'interruption utilisateur
router_builder.add_node("llm_call_1", llm_call_1)
router_builder.add_node("llm_call_2", llm_call_2)
router_builder.add_node("llm_call_3", llm_call_3)
router_builder.add_node("llm_call_router", llm_call_router)

# Add edges to connect nodes
router_builder.add_edge(START, "llm_call_router")
router_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    {  # Name returned by route_decision : Name of next node to visit
        "user_interrupt": "user_interrupt",  # Redirection vers le nœud d'interruption
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)

# Fonction conditionnelle pour décider si on continue après l'interruption
def check_user_approval(state: State):
    if state.get("user_approval", False):
        return "llm_call_1"  # Continuer vers llm_call_1 si l'utilisateur approuve
    else:
        return END  # Terminer le flux si l'utilisateur n'approuve pas

# Ajouter les arêtes conditionnelles pour le nœud d'interruption
router_builder.add_conditional_edges(
    "user_interrupt",
    check_user_approval,
    {
        "llm_call_1": "llm_call_1",
        END: END
    }
)

router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile workflow
router_workflow = router_builder.compile()


# Invoke
# Pour tester l'agent avec l'outil de génération de titres
def test_agent():
    logging.info("\n\n=== DÉBUT DU TEST DE L'AGENT AVEC L'OUTIL DE GÉNÉRATION DE TITRES ET INTERRUPTION UTILISATEUR ===")
    
    # Test avec une requête d'histoire
    test_input = "Raconte-moi une histoire sur les robots"
    logging.info(f"Test avec l'entrée: '{test_input}'")
    
    # Créer un checkpoint pour pouvoir reprendre l'exécution après l'interruption
    checkpoint = Checkpoint()
    
    # Première étape : jusqu'à l'interruption utilisateur
    state = router_workflow.invoke({"input": test_input}, checkpoint=checkpoint)
    
    # Afficher le message d'interruption
    logging.info(f"Message d'interruption: '{state['output']}'")
    
    # Dans un système réel, on attendrait ici l'entrée de l'utilisateur
    # Pour la démonstration, nous simulons une approbation
    
    # Continuer l'exécution avec l'approbation utilisateur
    final_state = router_workflow.invoke({"user_approval": True}, checkpoint=checkpoint)
    
    logging.info(f"Sortie finale: '{final_state['output']}'")
    logging.info("=== FIN DU TEST DE L'AGENT AVEC L'OUTIL DE GÉNÉRATION DE TITRES ET INTERRUPTION UTILISATEUR ===\n\n")
    return final_state

# Exécuter le test
test_result = test_agent()
print("Résultat final:", test_result["output"])
