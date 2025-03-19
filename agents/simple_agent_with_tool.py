"""
Un exemple simple d'agent LangGraph avec un outil d'addition.
"""

# Définition de l'outil add
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

# Définition d'un nœud qui utilise l'outil add
def math_node(state):
    """Node that performs addition using the add function."""
    # Extraire les nombres de l'entrée (dans un cas réel, un LLM le ferait)
    result = add(5, 10)
    return {"output": f"Le résultat de l'addition est: {result}"}

# Fonction principale
def main():
    print("Simulation d'un agent LangGraph avec outil add")
    
    # Simuler l'exécution du nœud
    state = {"input": "Calcule 5 + 10"}
    result = math_node(state)
    
    print(f"Entrée: {state['input']}")
    print(f"Sortie: {result['output']}")
    
    print("\nPour implémenter cet agent dans LangGraph complet:")
    print("1. Définir l'outil avec @tool de langchain_core.tools")
    print("2. Lier l'outil au LLM avec llm.bind_tools([add])")
    print("3. Créer un nœud qui utilise le LLM avec les outils")
    print("4. Ajouter ce nœud à un StateGraph")
    print("5. Définir les arêtes du graphe")
    print("6. Compiler et exécuter le graphe")

if __name__ == "__main__":
    main()
