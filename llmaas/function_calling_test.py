import requests
import json

def send_request(url_endpoint, payload):
    """Envoie une requête POST à l'endpoint spécifié et retourne la réponse."""
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url_endpoint, json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # Lève une exception pour les codes d'erreur HTTP
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Erreur lors de l'envoi de la requête :", e)
        return None

def tester_function_calling(url_endpoint, function_name, params):
    """Teste si l'endpoint supporte le function calling."""
    # Construction du payload
    payload = {
        "prompt": f"Calcule la somme de {params['a']} et {params['b']} en utilisant la fonction.",
        "function_call": True,
        "functions": [
            {
                "name": function_name,
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

    # Envoi de la requête
    resultat = send_request(url_endpoint, payload)
    if resultat and "function_call" in resultat:
        function_call = resultat["function_call"]
        
        # Validation du nom de la fonction
        if function_call.get("name") != function_name:
            print("Erreur : Mauvaise fonction appelée")
            return False
        
        # Validation des paramètres
        if not all(k in function_call.get("parameters", {}) for k in ("a", "b")):
            print("Erreur : Paramètres manquants")
            return False

        # Exécution de la fonction
        try:
            result = additionner(function_call["parameters"]["a"], function_call["parameters"]["b"])
            print(f"Résultat du calcul : {result}")
            return True
        except Exception as e:
            print(f"Erreur d'exécution : {e}")
            return False
    else:
        print("La réponse ne contient pas de données de function calling.")
        return False

# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python attack_endpoint.py <endpoint_url>")
        sys.exit(1)
    endpoint_url = sys.argv[1]
    params = {"a": 5, "b": 7}  # Paramètres à tester
    if tester_function_calling(endpoint_url, "additionner", params):
        print("[SUCCESS] Function calling opérationnel")
    else:
        print("[FAIL] Problème détecté")
