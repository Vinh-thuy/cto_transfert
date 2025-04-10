import sys
import traceback
import logging
sys.path.append('/Users/vinh/Documents/cto_transfert')

from prepare_sqlite_database import prepare_sqlite_database
from langgraph_trend_analysis_agent import TrendAnalysisAgent

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    try:
        # Chemin par défaut du fichier CSV
        csv_path = '/Users/vinh/Documents/cto_transfert/data_simulation.csv'
        
        # Vérifier si un chemin de fichier est passé en argument
        if len(sys.argv) > 1:
            csv_path = sys.argv[1]
        
        # Préparer la base de données (si ce n'est pas déjà fait)
        logger.info("Préparation de la base de données...")
        prepare_sqlite_database(csv_path)
        
        # Initialiser l'agent
        logger.info("Initialisation de l'agent d'analyse...")
        agent = TrendAnalysisAgent(target_column=['Caddies'], 
                                   additional_columns=['Passages', 'Vues', 'Conversion'])
        
        # Analyser uniquement Pierre
        logger.info("Début de l'analyse pour Pierre...")
        analysis = agent.analyze_user_trends("Pierre")
        
        print("\n" + "="*50)
        print(f"🔍 Analyse de Pierre")
        print("="*50)
        
        if analysis:
            print("\n📊 Description de la tendance :")
            print(analysis.get('trend_description', 'Aucune description disponible'))
            
            print("\n📈 Métriques globales :")
            print(f"- Indice de volatilité : {analysis.get('volatility_index', 'N/A')}")
            print(f"- Taux de croissance : {analysis.get('growth_rate', 'N/A')}%")
            
            print("\n🔢 Métriques détaillées :")
            for col in ['Caddies', 'Passages', 'Vues', 'Conversion']:
                print(f"\n{col} :")
                print(f"  - Volatilité : {analysis.get(f'{col}_volatility', 'N/A')}")
                print(f"  - Taux de croissance : {analysis.get(f'{col}_growth_rate', 'N/A')}%")
                print(f"  - Minimum : {analysis.get(f'{col}_min', 'N/A')}")
                print(f"  - Maximum : {analysis.get(f'{col}_max', 'N/A')}")
            
            print("\n🔑 Observations clés :")
            for obs in analysis.get('key_observations', []):
                print(f"- {obs}")
        else:
            logger.warning("Aucune analyse trouvée pour Pierre")
    
    except Exception as e:
        logger.error("Erreur lors de l'exécution :")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()
