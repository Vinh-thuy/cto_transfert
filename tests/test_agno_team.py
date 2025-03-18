import pytest
import pandas as pd
from agents.agno_team import (
    analyse_similitude_question, 
    get_indicateur_definition, 
    router_question,
    INDICATEURS_PILOTAGE,
    QUESTIONS_STRATEGIQUES
)

def test_get_indicateur_definition():
    """Test la récupération des définitions d'indicateurs"""
    # Test pour un indicateur existant
    assert get_indicateur_definition("EBITDA") == "Bénéfice avant intérêts, impôts, dépréciation et amortissement"
    
    # Test pour un indicateur non existant
    assert get_indicateur_definition("Indicateur Inexistant") == "Indicateur non référencé"

def test_router_question():
    """Test le routage des questions vers le bon agent"""
    # Questions d'indicateurs
    assert router_question("Quel est le calcul de l'EBITDA ?") == "indicateurs"
    assert router_question("Expliquez le Chiffre d'Affaires") == "indicateurs"
    
    # Questions stratégiques
    assert router_question("Comment améliorer notre entreprise ?") == "strategique"
    assert router_question("Quelles sont nos perspectives de croissance ?") == "strategique"

def test_analyse_similitude_question():
    """Test l'analyse de similitude des questions stratégiques"""
    # Question proche des questions stratégiques
    resultats = analyse_similitude_question("Comment développer notre entreprise ?")
    
    # Vérifier que des résultats sont retournés
    assert len(resultats) > 0
    
    # Vérifier la structure des résultats
    for resultat in resultats:
        assert 'question' in resultat
        assert 'categorie' in resultat
        assert 'score' in resultat
        assert 0 <= resultat['score'] <= 1

def test_questions_strategiques_dataframe():
    """Validation de la structure du DataFrame des questions stratégiques"""
    # Vérifier que le DataFrame n'est pas vide
    assert not QUESTIONS_STRATEGIQUES.empty
    
    # Vérifier les colonnes
    colonnes_attendues = ['question', 'categorie', 'mots_cles']
    for colonne in colonnes_attendues:
        assert colonne in QUESTIONS_STRATEGIQUES.columns
    
    # Vérifier la cohérence des données
    assert len(QUESTIONS_STRATEGIQUES) == len(QUESTIONS_STRATEGIQUES['categorie'])
    assert len(QUESTIONS_STRATEGIQUES) == len(QUESTIONS_STRATEGIQUES['mots_cles'])

def test_indicateurs_pilotage():
    """Validation du dictionnaire des indicateurs de pilotage"""
    # Vérifier que le dictionnaire n'est pas vide
    assert len(INDICATEURS_PILOTAGE) > 0
    
    # Vérifier que chaque indicateur a une définition non vide
    for indicateur, definition in INDICATEURS_PILOTAGE.items():
        assert isinstance(indicateur, str)
        assert isinstance(definition, str)
        assert len(definition) > 0

# Tests d'intégration
def test_integration_indicateurs():
    """Test d'intégration pour les questions d'indicateurs"""
    question = "Quel est le calcul de l'EBITDA ?"
    route = router_question(question)
    assert route == "indicateurs"
    
    definition = get_indicateur_definition("EBITDA")
    assert "Bénéfice" in definition

def test_integration_strategique():
    """Test d'intégration pour les questions stratégiques"""
    question = "Comment développer notre entreprise ?"
    route = router_question(question)
    assert route == "strategique"
    
    resultats = analyse_similitude_question(question)
    assert len(resultats) > 0
    assert any("croissance" in res['question'].lower() for res in resultats)
