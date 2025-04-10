import sqlite3
import pandas as pd
import logging
from typing import Dict, Any, Optional, Union, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrendAnalysisAgent:
    def __init__(self, 
                 db_path: str = 'data_analysis.db', 
                 table_name: str = 'data_table',
                 name_column: str = 'Name', 
                 date_column: str = 'Day', 
                 target_column: Union[str, List[str]] = 'Caddies',
                 additional_columns: Optional[List[str]] = None):
        """
        Initialiser l'agent d'analyse de tendances
        
        Args:
            db_path (str): Chemin vers la base de données SQLite
            table_name (str): Nom de la table dans la base de données
            name_column (str): Nom de la colonne contenant les noms
            date_column (str): Nom de la colonne contenant les dates
            target_column (str ou List[str]): Colonnes à analyser
            additional_columns (List[str], optionnel): Colonnes supplémentaires à inclure
        """
        self.db_path = db_path
        self.table_name = table_name
        self.name_column = name_column
        self.date_column = date_column
        
        # Gérer les colonnes cibles
        if isinstance(target_column, str):
            self.target_columns = [target_column]
        else:
            self.target_columns = target_column
        
        # Colonnes supplémentaires
        self.additional_columns = additional_columns or []
        
        self.llm = ChatOpenAI(
            model="gpt-4o-mini", 
            temperature=0.2, 
            max_tokens=1500
        )
        self.output_parser = JsonOutputParser()
        
        # Définir le prompt pour l'analyse
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """Tu es un analyste de données spécialisé dans l'analyse de tendances.
            Analyse les données suivantes de manière factuelle et technique.
            
            Données pour {name} entre {start_date} et {end_date}:
            {data_summary}
            
            Fournis une analyse technique au format JSON avec les clés suivantes:
            - trend_description: Description technique de la tendance
            - volatility_index: Un indice de volatilité global (0-100)
            - growth_rate: Taux de croissance global en pourcentage
            - key_observations: Liste des observations techniques
            """),
            ("human", "Analyse les données.")
        ])


        print('----------')
        print("""Tu es un analyste de données spécialisé dans l'analyse de tendances.
            Analyse les données suivantes de manière factuelle et technique.
            
            Données pour {name} entre {start_date} et {end_date}:
            {data_summary}
            
            Fournis une analyse technique au format JSON avec les clés suivantes:
            - trend_description: Description technique de la tendance
            - volatility_index: Un indice de volatilité global (0-100)
            - growth_rate: Taux de croissance global en pourcentage
            - key_observations: Liste des observations techniques
            """)
        
    def _calculate_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculer les métriques pour les colonnes cibles
        
        Args:
            df (pd.DataFrame): DataFrame à analyser
        
        Returns:
            Dict[str, Any]: Métriques calculées
        """
        metrics = {}
        
        for col in self.target_columns:
            if col not in df.columns:
                continue
            
            # Métriques par colonne
            metrics[f'{col}_volatility'] = df[col].std() / df[col].mean() * 100 if df[col].mean() != 0 else 0
            metrics[f'{col}_growth_rate'] = ((df[col].iloc[-1] - df[col].iloc[0]) / df[col].iloc[0]) * 100 if df[col].iloc[0] != 0 else 0
            metrics[f'{col}_min'] = df[col].min()
            metrics[f'{col}_max'] = df[col].max()
        
        # Métriques globales
        metrics['volatility_index'] = sum(v for k, v in metrics.items() if k.endswith('_volatility')) / len(self.target_columns)
        metrics['growth_rate'] = sum(v for k, v in metrics.items() if k.endswith('_growth_rate')) / len(self.target_columns)
        
        return metrics
        
    def analyze_user_trends(self, 
                             name: str, 
                             start_date: Optional[str] = None, 
                             end_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyser les tendances pour un utilisateur spécifique
        
        Args:
            name (str): Nom de l'utilisateur
            start_date (str, optionnel): Date de début de l'analyse
            end_date (str, optionnel): Date de fin de l'analyse
        
        Returns:
            Dict[str, Any]: Résultats de l'analyse
        """
        logger.info(f"Début de l'analyse pour {name}")
        
        try:
            # Connexion à la base de données
            conn = sqlite3.connect(self.db_path)
            
            # Construire la requête SQL dynamiquement
            query_conditions = [f"{self.name_column} = '{name}'"]
            
            if start_date:
                query_conditions.append(f"{self.date_column} >= '{start_date}'")
            
            if end_date:
                query_conditions.append(f"{self.date_column} <= '{end_date}'")
            
            # Sélectionner toutes les colonnes nécessaires
            select_columns = [self.date_column, self.name_column] + self.target_columns + self.additional_columns
            
            # Requête SQL pour récupérer les données
            query = f"""
            SELECT 
                {', '.join(select_columns)}
            FROM {self.table_name}
            WHERE {' AND '.join(query_conditions)}
            ORDER BY {self.date_column}
            """
            
            # Exécuter la requête
            df = pd.read_sql_query(query, conn)
            
            # Fermer la connexion
            conn.close()
            
            # Vérifier si des données sont disponibles
            if df.empty:
                logger.warning(f"Aucune donnée trouvée pour {name}")
                return {}
            
            # Calculer les métriques
            metrics = self._calculate_metrics(df[self.target_columns])

            print('---- DATA metrics ----')
            print(metrics)


            # Préparer les données pour le LLM
            data_summary = df.to_string()

            print('---- DATA SUMMARY ----')
            print(data_summary)
            
            # Préparer les données pour le LLM
            chain = self.prompt | self.llm | self.output_parser
            
            # Générer l'analyse
            analysis = chain.invoke({
                "name": name,
                "start_date": start_date or df[self.date_column].min(),
                "end_date": end_date or df[self.date_column].max(),
                "data_summary": data_summary,
                **metrics
            })
            
            # Ajouter les métriques détaillées à l'analyse
            analysis.update(metrics)
            
            return analysis
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse pour {name} : {e}")
            return {}

def main():
    logger.info("Préparation de la base de données...")
    from prepare_sqlite_database import prepare_sqlite_database
    prepare_sqlite_database()
    
    logger.info("Initialisation de l'agent d'analyse...")
    agent = TrendAnalysisAgent()
    
    logger.info("Début de l'analyse pour Pierre...")
    result = agent.analyze_user_trends('Pierre')
    print(result)

if __name__ == "__main__":
    main()
