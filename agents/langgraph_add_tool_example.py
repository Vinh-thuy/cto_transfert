"""
Exemple détaillé d'un agent LangGraph avec un outil d'addition.

Ce code est une démonstration conceptuelle qui montre comment un outil d'addition
serait intégré dans un agent LangGraph. Pour l'exécuter réellement, vous auriez besoin
des dépendances appropriées (langchain, langgraph, etc.).
"""

# Définition de l'outil add
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

# Simulation d'un LLM
class SimulatedLLM:
    def __init__(self):
        self.tools = {}
    
    def bind_tools(self, tools_list):
        """Simule la liaison des outils au LLM."""
        for tool in tools_list:
            self.tools[tool.__name__] = tool
        return self
    
    def invoke(self, prompt):
        """Simule l'invocation du LLM avec un prompt."""
        # Dans un cas réel, le LLM analyserait le prompt et déciderait quand utiliser les outils
        if "addition" in prompt.lower() or "+" in prompt:
            # Extraction simulée des nombres (dans un cas réel, le LLM le ferait)
            numbers = [int(s) for s in prompt.split() if s.isdigit()]
            if len(numbers) >= 2:
                result = self.tools["add"](numbers[0], numbers[1])
                return f"J'ai utilisé l'outil 'add' pour calculer {numbers[0]} + {numbers[1]} = {result}"
        return "Je ne peux pas traiter cette demande."

# Simulation d'un nœud de graphe LangGraph
def llm_call_1(state):
    """Simule un nœud LangGraph qui utilise un LLM avec des outils."""
    # Dans un cas réel, ce serait un vrai LLM avec des outils
    llm = SimulatedLLM()
    llm_with_tools = llm.bind_tools([add])
    
    result = llm_with_tools.invoke(state["input"])
    return {"output": result}

# Simulation du graphe LangGraph
def simulate_langgraph():
    """Simule l'exécution d'un graphe LangGraph."""
    print("=== Simulation d'un agent LangGraph avec outil add ===")
    
    # Simulation des états et transitions
    print("\n1. Initialisation de l'état")
    state = {"input": "Calcule l'addition de 42 + 58"}
    print(f"   État initial: {state}")
    
    print("\n2. Exécution du nœud llm_call_1 (qui utilise l'outil add)")
    new_state = llm_call_1(state)
    state.update(new_state)
    print(f"   État mis à jour: {state}")
    
    print("\n3. Résultat final")
    print(f"   Sortie: {state['output']}")
    
    print("\n=== Fin de la simulation ===")

# Explication de l'implémentation réelle
def explain_real_implementation():
    print("\n=== Comment implémenter cela avec LangGraph réel ===")
    print("""
Dans une implémentation réelle avec LangGraph:

1. Définition de l'outil:
   ```python
   from langchain_core.tools import tool
   
   @tool
   def add(a: int, b: int) -> int:
       \"\"\"Adds a and b.\"\"\"
       return a + b
   ```

2. Configuration du LLM avec l'outil:
   ```python
   from langchain_openai import ChatOpenAI
   
   llm = ChatOpenAI()
   tools = [add]
   llm_with_tools = llm.bind_tools(tools)
   ```

3. Définition du nœud dans le graphe:
   ```python
   def llm_call_1(state):
       \"\"\"Node that uses the add tool.\"\"\"
       result = llm_with_tools.invoke(state["input"])
       return {"output": result.content}
   ```

4. Construction du graphe:
   ```python
   from langgraph.graph import StateGraph, START, END
   
   builder = StateGraph()
   builder.add_node("llm_call_1", llm_call_1)
   builder.add_edge(START, "llm_call_1")
   builder.add_edge("llm_call_1", END)
   
   graph = builder.compile()
   ```

5. Exécution:
   ```python
   result = graph.invoke({"input": "Calcule 42 + 58"})
   print(result["output"])
   ```
""")

if __name__ == "__main__":
    simulate_langgraph()
    explain_real_implementation()
