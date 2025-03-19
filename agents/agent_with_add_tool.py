"""Agent avec outil d'addition - Version sans dépendances externes

Ce script simule un agent LangGraph avec un outil d'addition, sans utiliser
de dépendances externes. Il montre comment l'outil serait intégré dans un
agent réel, mais de manière simplifiée pour éviter les problèmes d'installation.
"""
from typing import Dict, Any

# Définition de l'outil add (sans décorateur @tool)
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

# Simulation d'un état
class State(dict):
    """Classe simple pour simuler un état dans LangGraph."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

# Fonction qui simule un nœud utilisant le tool add
def math_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node that performs addition using the add tool."""
    print(f"\nTraitement de l'entrée: {state['input']}")
    print("Utilisation de l'outil 'add'...")
    
    # Dans un cas réel, un LLM interpréterait l'entrée
    # Pour simplifier, on extrait directement les nombres
    try:
        # Extraction simplifiée des nombres (dans un cas réel, le LLM ferait mieux)
        parts = state['input'].replace('Calcule', '').replace('et', '+').strip().split('+')
        if len(parts) >= 2:
            a = int(parts[0].strip())
            b = int(parts[1].strip())
            result = add(a=a, b=b)
            return {"output": f"Le résultat de l'addition de {a} + {b} est: {result}"}
        else:
            return {"output": "Je n'ai pas pu identifier deux nombres à additionner."}
    except Exception as e:
        return {"output": f"Erreur lors du traitement: {str(e)}"}

# Simulation d'un graphe LangGraph
class SimpleGraph:
    """Classe simple pour simuler un graphe LangGraph."""
    def __init__(self):
        self.nodes = {}
    
    def add_node(self, name, func):
        """Ajoute un nœud au graphe."""
        self.nodes[name] = func
        return self
    
    def invoke(self, initial_state):
        """Simule l'exécution du graphe avec un état initial."""
        state = State(**initial_state)
        print("\n=== Exécution du graphe ===")
        print(f"État initial: {state}")
        
        # Exécution du nœud math_node
        if "math_node" in self.nodes:
            result = self.nodes["math_node"](state)
            state.update(result)
        
        print(f"État final: {state}")
        return state

# Simulation de l'exécution
if __name__ == "__main__":
    print("=== Simulation d'un agent LangGraph avec outil add ===\n")
    
    # Création du graphe
    print("1. Création du graphe avec le nœud math_node")
    graph = SimpleGraph()
    graph.add_node("math_node", math_node)
    
    # Exécution avec différentes entrées
    print("\n2. Test avec une addition simple")
    result1 = graph.invoke({"input": "Calcule 5 + 10"})
    
    print("\n3. Test avec une autre addition")
    result2 = graph.invoke({"input": "Calcule 42 et 58"})
    
    print("\n=== Fin de la simulation ===\n")
    print("Dans un agent LangGraph réel:")
    print("1. L'outil 'add' serait défini avec @tool de langchain_core.tools")
    print("2. Un LLM analyserait l'entrée et déciderait quand utiliser l'outil")
    print("3. Le graphe serait construit avec StateGraph de langgraph")
    print("4. Les nœuds et arêtes définiraient le flux d'exécution")
