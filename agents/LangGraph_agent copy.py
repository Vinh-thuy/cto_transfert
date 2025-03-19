from langchain_openai import ChatOpenAI
from typing_extensions import Literal
from langchain_core.messages import HumanMessage, SystemMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from pydantic import BaseModel, Field


# Charger les variables d'environnement
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = "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"
print("API Key:", openai_api_key)


llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, temperature=0)



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


# Nodes
def llm_call_1(state: State):
    """Write a story"""
    logging.info("Calling llm_call_1 for story generation.")
    system_prompt = "Vous êtes un écrivain créatif. Écrivez une histoire avec de l'humour noir en quelques lignes (2 max)basée sur :"
    full_prompt = f"{system_prompt} {state['input']}"
    result = llm.invoke(full_prompt)
    return {"output": result.content}


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
    logging.info(f"Routing input: {state['input']}")
    decision = router.invoke(
        [
            SystemMessage(
                content=router_prompt
            ),
            HumanMessage(content=state["input"]),
        ]
    )
    logging.info(f"Decision made: {decision.step}")
    return {"decision": decision.step}


# Conditional edge function to route to the appropriate node
def route_decision(state: State):
    # Return the node name you want to visit next
    if state["decision"] == "story":
        return "llm_call_1"
    elif state["decision"] == "joke":
        return "llm_call_2"
    elif state["decision"] == "chatbot":
        return "llm_call_3"


# Build workflow
router_builder = StateGraph(State)

# Add nodes
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
        "llm_call_1": "llm_call_1",
        "llm_call_2": "llm_call_2",
        "llm_call_3": "llm_call_3",
    },
)
router_builder.add_edge("llm_call_1", END)
router_builder.add_edge("llm_call_2", END)
router_builder.add_edge("llm_call_3", END)

# Compile workflow
router_workflow = router_builder.compile()


# Invoke
# state = router_workflow.invoke({"input": "Write me a poem about forest"})
# print(state["output"])