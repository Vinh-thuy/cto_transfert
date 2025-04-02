#!/bin/bash

# Script pour configurer l'environnement virtuel avec UV

# Vérifier si UV est installé
if ! command -v uv &> /dev/null; then
    echo "UV n'est pas installé. Installation en cours..."
    curl -sSf https://install.python-uv.org | python3
fi

# Créer l'environnement virtuel
echo "Création de l'environnement virtuel avec UV..."
uv venv .venv

# Activer l'environnement virtuel
echo "Pour activer l'environnement virtuel, exécutez:"
echo "source .venv/bin/activate"

# Installer les dépendances avec UV
echo "Installation des dépendances avec UV..."
uv pip install -r requirements.txt

echo "Configuration de l'environnement terminée!"
