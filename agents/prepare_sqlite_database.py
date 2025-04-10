import pandas as pd
import sqlite3
import logging
from typing import List, Optional, Union, Dict

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def infer_column_types(df: pd.DataFrame) -> dict:
    """
    Inférer les types de colonnes SQLite à partir d'un DataFrame pandas
    
    Args:
        df (pd.DataFrame): DataFrame source
    
    Returns:
        dict: Mapping des colonnes avec leurs types SQLite
    """
    column_types = {}
    for col, dtype in df.dtypes.items():
        if pd.api.types.is_datetime64_any_dtype(dtype):
            column_types[col] = 'DATE'
        elif pd.api.types.is_integer_dtype(dtype):
            column_types[col] = 'INTEGER'
        elif pd.api.types.is_float_dtype(dtype):
            column_types[col] = 'REAL'
        else:
            column_types[col] = 'TEXT'
    return column_types

def load_csv_to_sqlite(
    csv_path: str, 
    db_path: str = 'data_analysis.db', 
    table_name: str = 'data_table', 
    date_columns: Optional[Union[str, List[str]]] = None,
    date_format: str = '%d/%m/%Y',
    required_columns: Optional[List[str]] = None,
    optional_columns: Optional[List[str]] = None
) -> Optional[Dict[str, Union[str, int]]]:
    """
    Charger un fichier CSV dans une base de données SQLite avec flexibilité
    
    Args:
        csv_path (str): Chemin vers le fichier CSV
        db_path (str): Chemin de la base de données SQLite
        table_name (str): Nom de la table dans la base de données
        date_columns (str ou List[str], optionnel): Colonnes à parser comme des dates
        date_format (str, optionnel): Format des dates dans le CSV
        required_columns (List[str], optionnel): Colonnes requises
        optional_columns (List[str], optionnel): Colonnes optionnelles
        
    Returns:
        Optional[Dict[str, Union[str, int]]]: Dictionnaire avec les infos de la DB ou None si erreur
    """
    conn = None
    try:
        # Lire le fichier CSV
        df = pd.read_csv(csv_path, sep=';')
        
        # Convertir les dates si nécessaire
        if date_columns:
            if isinstance(date_columns, str):
                date_columns = [date_columns]
            
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], format=date_format, dayfirst=True)
        
        # Vérifier les colonnes requises
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Colonnes requises manquantes : {missing_columns}")
        
        # Ajouter des colonnes optionnelles si elles n'existent pas
        if optional_columns:
            for col in optional_columns:
                if col not in df.columns:
                    logger.warning(f"Colonne optionnelle '{col}' non trouvée. Ajout avec des valeurs nulles.")
                    df[col] = None
        
        # Convertir les dates au format ISO si nécessaire
        if date_columns:
            for col in date_columns:
                if col in df.columns:
                    df[col] = df[col].dt.strftime('%Y-%m-%d')
        
        # Connexion à la base de données SQLite
        conn = sqlite3.connect(db_path)
        
        # Supprimer la table si elle existe déjà
        conn.execute(f'DROP TABLE IF EXISTS {table_name}')
        
        # Inférer les types de colonnes
        column_types = infer_column_types(df)
        
        # Créer la table dynamiquement
        create_table_query = f"""
        CREATE TABLE {table_name} (
            {', '.join([f"{col} {type_}" for col, type_ in column_types.items()])}
        )
        """
        conn.execute(create_table_query)
        
        # Insérer les données
        df.to_sql(table_name, conn, if_exists='append', index=False)
        
        logger.info(f"Données chargées avec succès dans la table {table_name}")
        
        # Vérifier le nombre de lignes
        cursor = conn.cursor()
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        row_count = cursor.fetchone()[0]
        logger.info(f"Nombre de lignes dans la table : {row_count}")
        
        # Afficher le schéma de la table
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        logger.info("Schéma de la table :")
        for col in columns:
            logger.info(f"- {col[1]} ({col[2]})")
        
        # Retourner les informations de la base de données
        return {
            'db_path': db_path,
            'table_name': table_name,
            'row_count': row_count
        }
    
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données : {e}")
        return None
    
    finally:
        # Fermer la connexion
        if conn:
            conn.close()

def prepare_sqlite_database(
    csv_path: str = '/Users/vinh/Documents/cto_transfert/data_simulation_pierre.csv', 
    db_path: str = 'data_analysis.db', 
    table_name: str = 'data_table',
    date_columns: Optional[Union[str, List[str]]] = 'Day',
    required_columns: Optional[List[str]] = ['Name', 'Day', 'Caddies'],
    optional_columns: Optional[List[str]] = ['Passages', 'Vues', 'Conversion']
) -> Optional[Dict[str, Union[str, int]]]:
    """
    Wrapper pour préparer la base de données SQLite
    
    Args:
        csv_path (str): Chemin vers le fichier CSV
        db_path (str): Chemin de la base de données SQLite
        table_name (str): Nom de la table dans la base de données
        date_columns (str ou List[str], optionnel): Colonnes à parser comme des dates
        required_columns (List[str], optionnel): Colonnes requises
        optional_columns (List[str], optionnel): Colonnes optionnelles
        
    Returns:
        Optional[Dict[str, Union[str, int]]]: Dictionnaire avec les infos de la DB ou None si erreur
    """
    # Retourner le résultat de load_csv_to_sqlite
    return load_csv_to_sqlite(csv_path, db_path, table_name, date_columns, 
                               required_columns=required_columns, 
                               optional_columns=optional_columns)

def main():
    # Charger les données dans SQLite
    db_info = prepare_sqlite_database()
    if db_info:
        logger.info(f"Préparation de la base de données réussie: {db_info}")
    else:
        logger.error("Échec de la préparation de la base de données.")

if __name__ == "__main__":
    main()
