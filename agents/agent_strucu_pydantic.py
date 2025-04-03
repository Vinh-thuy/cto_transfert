import os
from dotenv import load_dotenv
import logging
from typing import TypedDict, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph

# Configuration logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Charger les variables d'environnement
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Modèle Pydantic pour l'extraction structurée de la ville
class CityExtraction(BaseModel):
    ville: str = Field(
        ..., 
        description="Nom de la ville extraite de la requête de l'utilisateur"
    )
    pays: Optional[str] = Field(
        default=None, 
        description="Pays de la ville (optionnel)"
    )

# Modèle Pydantic pour les activités
class CityActivities(BaseModel):
    activities: list[str] = Field(
        ..., 
        description="Liste des activités recommandées pour la ville"
    )
    type_activites: str = Field(
        default="touristique", 
        description="Type des activités (touristique, culturel, gastronomique, etc.)"
    )

# État pour le graphe
class CityActivityState(TypedDict):
    input: str
    raw_city: Optional[str]
    structured_city: Optional[CityExtraction]
    activities: Optional[CityActivities]

# LLM
llm = ChatOpenAI(
    model="gpt-4o-mini", 
    openai_api_key=openai_api_key, 
    temperature=0.3
)

# Node 1 : Extraction de la ville
def extract_city_node(state: CityActivityState):
    # Créer un parser Pydantic
    parser = PydanticOutputParser(pydantic_object=CityExtraction)

    # Prompt pour l'extraction de la ville
    city_extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un expert en extraction d'informations géographiques. 
        Ton rôle est d'identifier précisément la ville mentionnée dans une requête."""),
        ("human", """{input}

{format_instructions}""")
    ]).partial(
        format_instructions=parser.get_format_instructions()
    )

    # Chaîne pour l'extraction brute
    raw_extraction_chain = city_extraction_prompt | llm
    raw_city_response = raw_extraction_chain.invoke({"input": state['input']}).content
    
    # Afficher le texte brut
    print("Ville extraite (texte brut):", raw_city_response)

    # Extraction structurée
    structured_chain = city_extraction_prompt | llm | parser
    try:
        structured_city = structured_chain.invoke({"input": state['input']})
    except Exception as e:
        print(f"Erreur d'extraction structurée : {e}")
        structured_city = CityExtraction(ville="Inconnu", pays=None)

    return {
        "raw_city": raw_city_response,  # Texte brut
        "structured_city": structured_city  # JSON structuré
    }

# Node 2 : Recherche d'activités
def find_city_activities_node(state: CityActivityState):
    # Utiliser la ville du JSON structuré
    ville = state['structured_city'].ville if state['structured_city'] else None
    pays = state['structured_city'].pays if state['structured_city'] else None
    
    if not ville or ville == "Inconnu":
        return {"activities": None}

    # Créer un parser Pydantic
    parser = PydanticOutputParser(pydantic_object=CityActivities)

    # Prompt pour trouver des activités
    activities_prompt = ChatPromptTemplate.from_messages([
        ("system", """Tu es un guide touristique expert qui recommande des activités."""),
        ("human", """Recommande 5 activités incontournables à faire à {ville} au {pays}.
        
        {format_instructions}""")
    ]).partial(
        format_instructions=parser.get_format_instructions()
    )

    # Chaîne pour générer les activités
    activities_chain = activities_prompt | llm | parser
    
    try:
        activities = activities_chain.invoke({
            "ville": ville, 
            "pays": pays or ""
        })
    except Exception as e:
        print(f"Erreur de génération d'activités : {e}")
        activities = None

    return {
        "activities": activities
    }

# Construction du graphe
def create_city_activities_graph():
    workflow = StateGraph(CityActivityState)
    
    # Ajouter les nœuds
    workflow.add_node("extract_city", extract_city_node)
    workflow.add_node("find_activities", find_city_activities_node)
    
    # Définir les arêtes
    workflow.set_entry_point("extract_city")
    workflow.add_edge("extract_city", "find_activities")
    workflow.set_finish_point("find_activities")
    
    return workflow.compile()

# Fonction principale pour tester
def main():
    graph = create_city_activities_graph()
    
    # Exemple d'utilisation
    input_query = "Conseil moi des activités à Porto ?"
    result = graph.invoke({"input": input_query})
    
    # Affichage des résultats
    print("\nActivités recommandées:")
    if result.get('activities'):
        for activity in result['activities'].activities:
            print(f"- {activity}")

if __name__ == "__main__":
    main()
