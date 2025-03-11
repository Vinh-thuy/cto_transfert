import requests
import json


def additionner(a: float, b: float) -> float:
    """Calcule la somme de deux nombres"""
    return a + b


def tester_function_calling(url_endpoint):
    """
    Cette fonction envoie une requête au LLM pour demander l'exécution d'un appel de fonction.
    L'objectif est de vérifier si l'endpoint supporte la fonctionnalité de function calling.
    """
    # Préparation du payload pour activer le function calling
    payload = {
        "prompt": "Calcule la somme de 2 et 3 en utilisant une fonction.",
        "function_call": True,  # Indique au LLM d'utiliser le function calling
        "functions": [
            {
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
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    # Envoi de la requête POST à l'endpoint
    try:
        response = requests.post(url_endpoint, data=json.dumps(payload), headers=headers)
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

    # Vérification de la présence d'un appel de fonction dans la réponse
    if "function_call" in resultat:
        function_call = resultat["function_call"]
        
        # Validation du nom de fonction
        if function_call.get("name") != "additionner":
            print("Erreur : Mauvaise fonction appelée")
            return False
        
        # Validation des paramètres
        params = function_call.get("parameters", {})
        if not all(k in params for k in ("a", "b")):
            print("Erreur : Paramètres manquants")
            return False
        
        # Exécution réelle de la fonction
        try:
            result = additionner(params['a'], params['b'])
            print(f"Résultat du calcul : {result}")
            return True
        except Exception as e:
            print(f"Erreur d'exécution : {e}")
            return False
    else:
        print("La réponse ne contient pas de données de function calling.")
        return False


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python attack_endpoint.py <endpoint_url>")
        sys.exit(1)
    endpoint_url = sys.argv[1]
    if tester_function_calling(endpoint_url):
        print("[SUCCESS] Function calling opérationnel")
    else:
        print("[FAIL] Problème détecté")
