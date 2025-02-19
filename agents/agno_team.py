from textwrap import dedent

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

def tool_web_agent():
    message="""
    ## L'Espagne
    L'espagne a une surface de 3 000 000 000 000 metre carrés
    L'Espagne est un pays du sud-ouest de l'Europe, qui partage des fronti res avec la France, Gibraltar, le Maroc et le Portugal. La capitale de l'Espagne est Madrid. L'Espagne est connue pour ses villes historiques telles que Barcelone, Valence, S ville et Grenade, ainsi que pour ses plages, ses montagnes et ses vins.
    """
    return message

def tool_finance_agent():
    message="""
    ## L'Italie
    L'italie a été découverte en l'an 200 par des marsiens
    L'Italie est un pays situé  dans le sud de l'Europe, qui partage des fronti res avec la France, la Suisse, l'Autriche et la Slov nie. La capitale de l'Italie est Rome. L'Italie est connue pour ses sites historiques tels que le Colos ee, le Panth on, la Place Saint-Marc  Venise, ainsi que pour sa cuisine d lite, ses vins et son caf .
    """
    return message


web_agent = Agent(
    name="Web Agent",
    role="Search the web for information",
    model=OpenAIChat(id="gpt-4o-mini"),
    tools=[tool_web_agent],
    instructions=["Always include sources"],
    expected_output=dedent("""\
    ## {title}

    {Answer to the user's question}
    """),
    # This will make the agent respond directly to the user, rather than through the team leader.
    respond_directly=True,
    markdown=True,
)


finance_agent = Agent(
    name="Finance Agent",
    role="Get financial data",
    model=OpenAIChat(id="gpt-4o"),
    tools=[tool_finance_agent],
    instructions=["Use tables to display data"],
    expected_output=dedent("""\
    ## {title}

    {Answer to the user's question}
    """),
    # This will make the agent respond directly to the user, rather than through the team leader.
    respond_directly=True,
    markdown=True,
)

agent_team = Agent(
    team=[web_agent, finance_agent],
    instructions=["Always include sources", "Use tables to display data"],
    markdown=True,
    debug_mode=True,
)

agent_team.print_response(
    "Summarize analyst recommendations and share the latest news for NVDA", stream=True
)

import os
import pandas as pd
import numpy as np
from textwrap import dedent
from typing import List, Dict, Any

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.team import Team

# Initialisation des données de base
INDICATEURS_PILOTAGE = {
    "Chiffre d'Affaires": "Montant total des ventes de l'entreprise sur une période donnée",
    "Marge Brute": "Différence entre le chiffre d'affaires et le coût des marchandises vendues",
    "EBITDA": "Bénéfice avant intérêts, impôts, dépréciation et amortissement",
    "Taux de Rotation des Stocks": "Mesure de l'efficacité de la gestion des stocks",
    "Délai Moyen de Paiement": "Temps moyen entre la facturation et le paiement des clients"
}

QUESTIONS_STRATEGIQUES = pd.DataFrame({
    'question': [
        "Comment améliorer notre rentabilité ?",
        "Quelles sont nos perspectives de croissance ?",
        "Comment optimiser notre chaîne de valeur ?",
        "Stratégies de réduction des coûts",
        "Opportunités de diversification du portefeuille"
    ],
    'categorie': [
        'Performance Financière', 
        'Développement Stratégique', 
        'Efficacité Opérationnelle', 
        'Gestion Financière', 
        'Innovation et Diversification'
    ],
    'mots_cles': [
        ['rentabilité', 'performance', 'profit'],
        ['croissance', 'expansion', 'marché'],
        ['chaîne de valeur', 'processus', 'efficacité'],
        ['coûts', 'réduction', 'optimisation'],
        ['diversification', 'innovation', 'nouveaux marchés']
    ]
})

def analyse_similitude_question(question_utilisateur: str, questions_reference: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Analyse la similitude entre la question de l'utilisateur et les questions stratégiques
    
    Args:
        question_utilisateur (str): Question posée par l'utilisateur
        questions_reference (pd.DataFrame): DataFrame des questions de référence
    
    Returns:
        List[Dict[str, Any]]: Liste des questions similaires avec leur score de similitude
    """
    # Utilisation d'OpenAI pour l'analyse sémantique
    model = OpenAIChat(id="gpt-4o-mini")
    
    resultats = []
    for index, row in questions_reference.iterrows():
        # Analyse sémantique avec GPT
        prompt = f"""
        Compare ces deux phrases et donne un score de similitude de 0 à 1 :
        Question 1: {question_utilisateur}
        Question 2: {row['question']}
        
        Critères d'évaluation :
        - Thématique générale
        - Intention stratégique
        - Mots-clés communs
        
        Réponds uniquement avec un nombre entre 0 et 1.
        """
        
        try:
            score = float(model.generate(prompt).strip())
            if score > 0.5:  # Seuil de similitude
                resultats.append({
                    'question': row['question'],
                    'categorie': row['categorie'],
                    'score': score
                })
        except Exception as e:
            print(f"Erreur lors de l'analyse : {e}")
    
    # Trier par score décroissant
    return sorted(resultats, key=lambda x: x['score'], reverse=True)

# Agent de gestion des indicateurs de pilotage
indicateurs_agent = Agent(
    name="Agent Indicateurs",
    role="Fournir des définitions précises sur les indicateurs de pilotage",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Répondre de manière précise et concise",
        "Utiliser un langage clair et professionnel",
        "Si l'indicateur n'est pas connu, le signaler explicitement"
    ],
    tools=[
        lambda indicateur: INDICATEURS_PILOTAGE.get(indicateur, "Indicateur non référencé")
    ],
    expected_output=dedent("""\
    ## {indicateur}

    {définition}

    ### Recommandations
    {insights_optionnels}
    """),
    respond_directly=True,
    markdown=True
)

# Agent de questions stratégiques
strategique_agent = Agent(
    name="Agent Stratégique",
    role="Analyser et proposer des perspectives stratégiques",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Identifier les questions stratégiques pertinentes",
        "Fournir une analyse approfondie",
        "Proposer des pistes de réflexion concrètes"
    ],
    tools=[
        lambda question: analyse_similitude_question(question, QUESTIONS_STRATEGIQUES)
    ],
    expected_output=dedent("""\
    ## Analyse Stratégique

    ### Questions Similaires Identifiées
    {questions_similaires}

    ### Recommandations Stratégiques
    {recommandations}
    """),
    respond_directly=True,
    markdown=True
)

# Agent Routeur
routeur_agent = Agent(
    name="Agent Routeur",
    role="Orienter la question vers l'agent approprié",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Déterminer la nature de la question",
        "Choisir l'agent le plus adapté"
    ],
    tools=[
        lambda question: (
            "indicateurs" if any(ind.lower() in question.lower() for ind in INDICATEURS_PILOTAGE.keys())
            else "strategique"
        )
    ],
    expected_output=dedent("""\
    ## Orientation de la Question

    Agent cible : {agent_cible}
    Justification : {explication}
    """),
    respond_directly=True,
    markdown=True
)

# Création de l'équipe
strategic_team = Team(
    name="Équipe Stratégique et Pilotage",
    agents=[indicateurs_agent, strategique_agent, routeur_agent],
    leader_model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Collaborer pour fournir les meilleures informations",
        "Utiliser l'agent routeur pour orienter précisément les questions"
    ]
)

# Exemple d'utilisation
if __name__ == "__main__":
    # Test de routage et d'analyse
    question1 = "Quel est le calcul de l'EBITDA ?"
    question2 = "Comment développer notre entreprise sur de nouveaux marchés ?"
    
    print("Question 1 (Indicateur) :", strategic_team.run(question1))
    print("\n---\n")
    print("Question 2 (Stratégique) :", strategic_team.run(question2))