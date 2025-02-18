import os
import sys
import importlib.util

# Débogage des imports
print("Python Path:", sys.path)
spec = importlib.util.find_spec("whoosh")
print("Whoosh location:", spec.origin if spec else "Not found")

# Importer explicitement les modules
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser, FuzzyTermPlugin

# Créer un répertoire pour l'index s'il n'existe pas
INDEX_DIR = "product_index"
if not os.path.exists(INDEX_DIR):
    os.mkdir(INDEX_DIR)

# Définir un schéma pour indexer les produits
schema = Schema(
    name=TEXT(stored=True),  # Le nom du produit avec recherche textuelle
    code=ID(stored=True)     # Un code unique pour le produit
)

# Créer l'index
ix = index.create_in(INDEX_DIR, schema)

# Ajouter des produits à l'index
writer = ix.writer()
writer.add_document(name="iPhone 14 Pro", code="IP14P")
writer.add_document(name="MacBook Air M2", code="MBA2")
writer.add_document(name="iPad Pro", code="IPP")
writer.add_document(name="Apple Watch Series 9", code="AWS9")
writer.commit()

# Fonction pour rechercher des produits avec tolérance aux fautes de frappe
def search_product(query_text):
    with ix.searcher() as searcher:
        # Configurer le parser de requête avec le plugin de recherche approximative
        parser = QueryParser("name", schema)
        parser.add_plugin(FuzzyTermPlugin())
        
        # Créer une requête avec une tolérance de 1 caractère
        query = parser.parse(f"{query_text}~1")
        
        # Effectuer la recherche
        results = searcher.search(query)
        
        # Afficher les résultats
        print(f"Recherche de : {query_text}")
        print(f"Nombre de résultats : {len(results)}")
        for result in results:
            print(f"Produit trouvé : {result['name']} (Code: {result['code']})")

# Démonstration de recherches avec fautes de frappe
def main():
    print("Démonstration de la recherche approximative avec Whoosh\n")
    
    # Recherches avec différentes variations
    search_product("iphone")       # Recherche standard
    print("\n")
    search_product("iphon")        # Avec une faute de frappe
    print("\n")
    search_product("macbuk")       # Avec une faute de frappe significative
    print("\n")
    search_product("aple wach")    # Avec plusieurs fautes de frappe

if __name__ == "__main__":
    main()
