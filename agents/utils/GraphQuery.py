import os

def getTopic_Subtopic_all(f, nodes_df, node_table_tlm_df):
    # Filtrer les nodes avec des topics
    nodes_df = nodes_df[nodes_df['topic'].notna()]
    
    for topic_row in nodes_df.topic.unique():
        # Nettoyer le nom du topic
        topic_row_clean = "".join([i for i in topic_row if not i.isdigit()]).replace("-", "").lstrip()
        
        # Compter les sous-topics (à implémenter si nécessaire)
        nb_topic, nb_sub_topic = countTopic_Subtopic_Report(nodes_df[nodes_df.topic == topic_row])
        
        if nb_sub_topic != 0:
            # Écrire l'en-tête du topic
            f.write(f'# Le Topic "{topic_row_clean}" dispose des Sub Topic suivantes :\n')
            
            # Filtrer les nodes avec des sous-topics
            nodes_df = nodes_df[nodes_df['sub_topic'].notna()]
            
            for sub_topic_row in nodes_df[nodes_df.topic == topic_row].sub_topic.unique():
                # Nettoyer le nom du sous-topic
                sub_topic_row_clean = "".join([i for i in sub_topic_row if not i.isdigit()]).replace("-", "").lstrip()
                
                # Écrire le sous-topic
                f.write(f'- {sub_topic_row_clean}\n')
                
                # Écrire l'en-tête des questions du sous-topic
                f.write(f'## Le Sub Topic "{sub_topic_row_clean}" est associé, relié, traitée par les questions suivantes :\n')
                
                # Écrire les questions du sous-topic
                for question in nodes_df[(nodes_df['topic'].notna()) & (nodes_df.sub_topic == sub_topic_row)].name:
                    f.write(f'  * {question}\n')
        else:
            # Pour les topics sans sous-topics
            f.write(f'# Le Topic "{topic_row_clean}" est associé, relié, traitée par les questions suivantes :\n')
            
            for question in node_table_tlm_df[node_table_tlm_df.topic == topic_row].name:
                f.write(f'* La question "{question}" est associée au Topic "{topic_row_clean}"\n')

def getTopic_Subtopic(nodes_df, node_focus, source="Report"):
    nodes_df = nodes_df[nodes_df['topic'].notna()]
    
    # Fonction à compléter selon les besoins
    pass

def countTopic_Subtopic_Report(df):
    """
    Fonction de comptage des topics et sous-topics.
    À implémenter selon les besoins spécifiques.
    """
    nb_topic = len(df['topic'].unique())
    nb_sub_topic = len(df[df['sub_topic'].notna()]['sub_topic'].unique())
    return nb_topic, nb_sub_topic

# Exemple d'utilisation
if __name__ == "__main__":
    import pandas as pd
    
    # Exemple de DataFrames (à remplacer par vos données réelles)
    nodes_df = pd.DataFrame({
        'topic': ['Topic1', 'Topic1', 'Topic2'],
        'sub_topic': ['SubTopic1A', 'SubTopic1B', None],
        'name': ['Question1', 'Question2', 'Question3']
    })
    
    node_table_tlm_df = pd.DataFrame({
        'topic': ['Topic2'],
        'name': ['Question4']
    })
    
    # Générer le rapport
    with open('topics_report.txt', 'w', encoding='utf-8') as f:
        getTopic_Subtopic_all(f, nodes_df, node_table_tlm_df)
    print("Rapport généré dans topics_report.txt")
