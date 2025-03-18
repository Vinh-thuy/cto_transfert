import ollama

def additionner(a: int, b: int) -> int:
    """
    Additionne deux nombres entiers.

    Args:
        a (int): Le premier nombre entier.
        b (int): Le second nombre entier.

    Returns:
        int: La somme des deux nombres.
    """
    return a + b

# Utilisation du modèle Llama 3.1 avec la fonction d'addition
response = ollama.chat(
    model='smollm2:135m',
    messages=[{'role': 'user', 'content': 'Quelle est la somme de 5 et 7 ?'}],
    tools=[additionner],  # Référence directe à la fonction
)

# Traitement de la réponse pour appeler la fonction appropriée
outils_disponibles = {
    'additionner': additionner,
}

for outil in response['message']['tool_calls'] or []:
    fonction_a_appeler = outils_disponibles.get(outil['function']['name'])
    if fonction_a_appeler:
        resultat = fonction_a_appeler(**outil['function']['arguments'])
        print('Résultat de la fonction :', resultat)
    else:
        print('Fonction non trouvée :', outil['function']['name'])