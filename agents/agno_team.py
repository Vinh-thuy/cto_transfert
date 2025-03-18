import os
import pandas as pd
import numpy as np
from textwrap import dedent
from typing import List, Dict, Any

from agno.agent import Agent
from agno.models.openai import OpenAIChat

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

def analyse_similitude_question(question_utilisateur: str) -> List[Dict[str, Any]]:
    """
    Analyse la similitude entre la question de l'utilisateur et les questions stratégiques
    """
    model = OpenAIChat(id="gpt-4o-mini")
    
    resultats = []
    for index, row in QUESTIONS_STRATEGIQUES.iterrows():
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

def get_indicateur_definition(indicateur: str) -> str:
    """Récupère la définition d'un indicateur"""
    return INDICATEURS_PILOTAGE.get(indicateur, "Indicateur non référencé")

def router_question(question: str) -> str:
    """Détermine l'agent cible en fonction de la question"""
    return "indicateurs" if any(ind.lower() in question.lower() for ind in INDICATEURS_PILOTAGE.keys()) else "strategique"

# Agent Indicateurs
indicateurs_agent = Agent(
    name="Agent Indicateurs",
    role="Fournir des définitions précises sur les indicateurs de pilotage",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Répondre de manière précise et concise",
        "Utiliser un langage clair et professionnel",
        "Si l'indicateur n'est pas connu, le signaler explicitement"
    ],
    tools=[get_indicateur_definition],
    expected_output=dedent("""\
    ## {indicateur}

    {définition}

    ### Recommandations
    {insights_optionnels}
    """),
    respond_directly=True,
    markdown=True
)

# Agent Stratégique
strategique_agent = Agent(
    name="Agent Stratégique",
    role="Analyser et proposer des perspectives stratégiques",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Identifier les questions stratégiques pertinentes",
        "Fournir une analyse approfondie",
        "Proposer des pistes de réflexion concrètes"
    ],
    tools=[analyse_similitude_question],
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
    tools=[router_question],
    expected_output=dedent("""\
    ## Orientation de la Question

    Agent cible : {agent_cible}
    Justification : {explication}
    """),
    respond_directly=True,
    markdown=True
)

# Éditeur (Team Leader)
strategic_editor = Agent(
    name="Éditeur Stratégique",
    team=[indicateurs_agent, strategique_agent, routeur_agent],
    description="Coordonne l'analyse des questions stratégiques et des indicateurs",
    model=OpenAIChat(id="gpt-4o-mini"),
    instructions=[
        "Utiliser l'agent routeur pour orienter précisément les questions",
        "Collaborer avec les agents pour fournir les meilleures informations",
        "Synthétiser et présenter clairement les résultats"
    ],
    markdown=True,
    debug_mode=True,
)

# Exemple d'utilisation
if __name__ == "__main__":
    # Test de routage et d'analyse
    question1 = "Quel est le calcul de l'EBITDA ?"
    question2 = "Comment développer notre entreprise sur de nouveaux marchés ?"
    
    print("Question 1 (Indicateur) :", strategic_editor.print_response(question1))
    print("\n---\n")
    print("Question 2 (Stratégique) :", strategic_editor.print_response(question2))