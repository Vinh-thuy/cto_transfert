import json
from openai import OpenAI


def additionner(a: float, b: float) -> float:
    """Calcule la somme de deux nombres"""
    return a + b


def tester_function_calling(api_key, base_url=None, model="gpt-3.5-turbo"):
    """
    Cette fonction utilise le client OpenAI pour tester le function calling.
    L'objectif est de vérifier si l'endpoint supporte la fonctionnalité de function calling.
    """
    try:
        # Initialisation du client OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Définition de la fonction à appeler
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "additionner",
                    "description": "Calcule la somme de deux nombres.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number"},
                            "b": {"type": "number"}
                        },
                        "required": ["a", "b"]
                    }
                }
            }
        ]
        
        # Envoi de la requête via le client OpenAI
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Calcule la somme de 2 et 3 en utilisant une fonction."}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        # Analyse de la réponse
        message = response.choices[0].message
        
        # Vérification de la présence d'un appel de fonction
        if message.tool_calls:
            tool_call = message.tool_calls[0]
            
            # Vérification du type et du nom de la fonction
            if tool_call.type == "function" and tool_call.function.name == "additionner":
                # Extraction et validation des paramètres
                try:
                    params = json.loads(tool_call.function.arguments)
                    if not all(k in params for k in ("a", "b")):
                        print("Erreur : Paramètres manquants")
                        return False
                    
                    # Exécution réelle de la fonction
                    try:
                        result = additionner(float(params['a']), float(params['b']))
                        print(f"Résultat du calcul : {result}")
                        return True
                    except Exception as e:
                        print(f"Erreur d'exécution : {e}")
                        return False
                except json.JSONDecodeError:
                    print("Arguments de fonction invalides")
                    return False
            else:
                print(f"Mauvaise fonction appelée : {tool_call.function.name}")
                return False
        else:
            print("La réponse ne contient pas d'appel de fonction")
            return False
            
    except Exception as e:
        print(f"Erreur lors de l'appel au client OpenAI : {e}")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python attack_endpoint.py <api_key> [base_url] [model]")
        sys.exit(1)
    
    api_key = sys.argv[1]
    base_url = sys.argv[2] if len(sys.argv) > 2 else None
    model = sys.argv[3] if len(sys.argv) > 3 else "gpt-3.5-turbo"
    
    if tester_function_calling(api_key, base_url, model):
        print("[SUCCESS] Function calling opérationnel")
    else:
        print("[FAIL] Problème détecté")