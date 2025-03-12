import requests
import json


def additionner(a: float, b: float) -> float:
    """Calcule la somme de deux nombres"""
    return a + b


def tester_function_calling(url_endpoint, api_key=None, model="gpt-3.5-turbo"):
    """
    Cette fonction envoie une requête au LLM pour demander l'exécution d'un appel de fonction.
    L'objectif est de vérifier si l'endpoint supporte la fonctionnalité de function calling.
    Compatible avec l'API OpenAI.
    """
    # Préparation du payload pour activer le function calling (format OpenAI)
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Calcule la somme de 2 et 3 en utilisant une fonction."}
        ],
        "tools": [
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
        ],
        "tool_choice": "auto"  # Laisse le modèle décider s'il doit utiliser la fonction
    }
    
    # Configuration des headers avec authentification si nécessaire
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Envoi de la requête POST à l'endpoint
    try:
        response = requests.post(url_endpoint, data=json.dumps(payload), headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print("Erreur lors de l'envoi de la requête :", e)
        return False

    if response.status_code != 200:
        print("Erreur HTTP", response.status_code, response.text)
        return False

    try:
        resultat = response.json()
    except json.JSONDecodeError:
        print("La réponse n'est pas au format JSON :", response.text)
        return False

    # Vérification de la présence d'un appel de fonction dans la réponse (format OpenAI)
    if "choices" in resultat and len(resultat["choices"]) > 0:
        choice = resultat["choices"][0]
        if "message" in choice and "tool_calls" in choice["message"]:
            tool_calls = choice["message"]["tool_calls"]
            
            # Vérifie s'il y a au moins un appel de fonction
            if len(tool_calls) > 0 and tool_calls[0]["type"] == "function":
                function_call = tool_calls[0]["function"]
                
                # Validation du nom de fonction
                if function_call.get("name") != "additionner":
                    print("Erreur : Mauvaise fonction appelée")
                    return False
                
                # Validation des paramètres
                try:
                    params = json.loads(function_call.get("arguments", "{}"))
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
    
    print("La réponse ne contient pas de données de function calling.")
    return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python attack_endpoint.py <endpoint_url> [api_key]")
        sys.exit(1)
    
    endpoint_url = sys.argv[1]
    api_key = sys.argv[2] if len(sys.argv) > 2 else None
    
    if tester_function_calling(endpoint_url, api_key):
        print("[SUCCESS] Function calling opérationnel")
    else:
        print("[FAIL] Problème détecté")
