import pandas as pd
import numpy as np
from typing import Dict, List
import os
import requests  # Tu peux remplacer par httpx si besoin

# Exemple d'appel :
# response = openai_llm_complete(
#     prompt="Ton prompt ici",
#     api_key="TA_CLE_API",
#     base_url="https://api.openai.com/v1",
#     model="gpt-3.5-turbo"
# )

def openai_llm_complete(prompt, api_key, base_url="https://api.openai.com/v1", model="gpt-3.5-turbo", http_client=None, **kwargs):
    """
    Fonction simple pour interroger un LLM compatible OpenAI (paramétrable : base_url, api_key, http_client).
    """
    client = http_client or requests
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        **kwargs
    }
    response = client.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
    return result['choices'][0]['message']['content']



def generate_prompt(metrics_dict: dict, dataset_name: str) -> str:
    """Génère un prompt LLM pour un dataset donné"""
    return f"""
En tant qu’agent d’analyse financière IT sur « {dataset_name} », vous disposez des indicateurs clés par catégorie :

{metrics_dict}

Analysez ces chiffres en 4 axes complémentaires :
1. Volatilité (écart‑type, %B Bollinger)
2. Momentum (pente MA07, ADX MA07)
3. Risque (max drawdown cumul quotidien)
4. Saisonnalité (croissance MoM cumul mensuel)

Concentrez‑vous d’abord sur la volatilité, puis le momentum, le risque et enfin la saisonnalité.
Terminez par un résumé exécutif en 3 points clés.
"""


def generate_global_prompt(all_metrics: dict, syntheses_axe: dict) -> str:
    """
    Génère un prompt LLM pour une analyse globale multi-axes.
    Le LLM dispose pour chaque axe (incidents, changes, MTTR, SLA, incidents externes) d'une synthèse structurée contenant :
      - Les tendances majeures, pics, ruptures, anomalies, avec pour chaque événement la date et la valeur associée
      - Les risques/alertes
      - 2 à 3 recommandations concrètes
    Toutes les synthèses sont formatées en bullet points avec tag ([INCIDENT], [CHANGE], [MTTR], [SLA], [EXTERNE]).

    Spécificité incidents :
    - Les incidents P1/P2 sont critiques, rares, signaux forts : leur analyse doit être pondérée fortement, chaque événement est important.
    - Les incidents P3/P4/P5 sont beaucoup plus fréquents, leur volumétrie génère du bruit de fond : l'analyse doit se concentrer sur les tendances de masse et la détection de signaux faibles, sans confondre leur bruit avec les alertes P1/P2.
    - Demande au LLM de traiter séparément ces deux familles, de pondérer leur impact, et de synthétiser les signaux faibles des P3-P5 distinctement des alertes P1-P2.

    Objectif :
    - Croiser et combiner ces informations pour fournir une vision globale et transversale de la situation IT
    - Détecter toute corrélation ou causalité entre axes (ex : pics d'incidents et pics de changes, dérives MTTR liées à des incidents, etc.)
    - Synthétiser les 5 points clés à retenir pour le CTO
    - Proposer les recommandations prioritaires à mettre en œuvre

    Format attendu :
    1. Synthèse globale structurée (overview)
    2. Analyse séparée incidents P1/P2 (signaux forts) vs P3/P4/P5 (volumétrie, bruit, signaux faibles)
    3. Corrélations ou causalités détectées entre axes (avec dates/valeurs)
    4. 5 points clés pour le CTO
    5. Recommandations prioritaires
    """
    syntheses_txt = "\n".join([f"== {axe.upper()} ==\n{synth}" for axe, synth in syntheses_axe.items()])
    prompt = f"""
Vous êtes un expert analyste IT. Vous disposez des synthèses suivantes par axe d'analyse :

{syntheses_txt}

Votre mission :
- 1. Analysez l'ensemble des axes pour produire une vision globale de la situation IT.
- 2. Pour les incidents, traitez séparément les P1/P2 (critiques, signaux forts) et les P3/P4/P5 (volumétrie, bruit, signaux faibles). Pondérez l'importance de chaque famille selon leur impact réel sur le service.
- 3. Identifiez toute corrélation ou causalité entre incidents, changes, MTTR, SLA et incidents externes (en vous appuyant sur les dates et valeurs associées à chaque événement).
- 4. Dégagez les 5 points clés à retenir pour le CTO.
- 5. Proposez les recommandations prioritaires à mettre en œuvre.

Format attendu :
[SYNTHÈSE GLOBALE] ...
[INCIDENTS P1/P2] ...
[INCIDENTS P3/P4/P5] ...
[CORRÉLATIONS] ...
[POINTS CLÉS] ...
[RECOMMANDATIONS] ...
"""
    return prompt


# Axe 1 : analyse des pics de changements et du comportement des incidents sur ces pics
# Axe 1 : Analyse des pics d'incidents critiques
# ------------------------------------------------------------
# Objectif : Identifier les pics d'incidents critiques pour chaque entité (Pole),
# détecter les périodes de surcharge ou de rupture opérationnelle.
# Méthode :
# - Calcul du seuil de pic (mean + z_thresh*std)
# - Détection des jours où ce seuil est dépassé
# - Calcul du nombre de pics, de la moyenne sur ces pics et de la moyenne globale
# Valeur ajoutée : Permet d'anticiper les périodes à risque et d'ajuster les ressources.
# ------------------------------------------------------------
def compute_peak_peaks_analysis(df_inc: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Pour chaque entité (Pole), identifie les jours où les incidents dépassent mean+z_thresh*std,
    calcule le nombre de pics, la moyenne d'incidents sur ces pics et la moyenne globale d'incidents.
    """
    results = []
    for pole in df_inc['Pole'].unique():
        s_inc = df_inc[df_inc['Pole']==pole].set_index('Day')['Value'].sort_index()
        mu, sigma = s_inc.mean(), s_inc.std()
        peaks = s_inc[s_inc > mu + z_thresh * sigma].index
        avg_inc_peaks = s_inc.reindex(peaks).mean()
        overall_avg_inc = s_inc.mean()
        results.append({'Pole': pole, 'PeakCount': len(peaks), 'AvgIncOnPeaks': avg_inc_peaks, 'OverallAvgInc': overall_avg_inc})
    return pd.DataFrame(results)

# Axe 2 : Analyse unitaire des Emergency Changes
# ------------------------------------------------------------
# Objectif : Analyser la volumétrie et la fréquence des Emergency Changes pour chaque entité, sans aucun croisement avec les incidents.
# Méthode :
# - Calcul du nombre de jours avec Emergency Changes (value>0) par entité
# Valeur ajoutée : Permet de suivre l'activité d'urgence et d'alerter sur une sur-utilisation potentielle des Emergency Changes.
# ------------------------------------------------------------
def compute_change_peaks_analysis(df_chg: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Pour chaque entité (Pole), identifie les jours où les changes dépassent mean+z_thresh*std,
    calcule le nombre de pics, la moyenne sur ces pics et la moyenne globale.
    """
    results = []
    for pole in df_chg['Pole'].unique():
        s_chg = df_chg[df_chg['Pole']==pole].set_index('Day')['Value'].sort_index()
        mu, sigma = s_chg.mean(), s_chg.std()
        peaks = s_chg[s_chg > mu + z_thresh * sigma].index
        avg_chg_peaks = s_chg.reindex(peaks).mean()
        overall_avg_chg = s_chg.mean()
        results.append({'Pole': pole, 'PeakCount': len(peaks), 'AvgChgOnPeaks': avg_chg_peaks, 'OverallAvgChg': overall_avg_chg})
    return pd.DataFrame(results)
# Axe 3 : Analyse unitaire des Standard Changes
# ------------------------------------------------------------
# Objectif : Analyser la volumétrie, la dynamique et les pics de Standard Changes pour chaque entité, sans aucun croisement avec les incidents.
# Méthode :
# - Calcul du nombre total de Standard Changes
# - Détection de pics (jours avec valeur > moyenne + 2*écart-type)
# - Calcul de la moyenne sur les pics et de la moyenne globale
# Valeur ajoutée : Permet de suivre l'activité normale de changement et d'alerter sur des périodes d'activité inhabituelle.
# ------------------------------------------------------------
def compute_standard_change_stats(df_std: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Pour chaque entité, calcule le nombre total de Standard Changes, le nombre de pics (jours > moyenne + z_thresh*std), la moyenne sur les pics et la moyenne globale.
    """
    results = []
    for pole in df_std['Pole'].unique():
        s_std = df_std[df_std['Pole']==pole].set_index('Day')['Value'].sort_index()
        mu, sigma = s_std.mean(), s_std.std()
        peaks = s_std[s_std > mu + z_thresh * sigma].index
        avg_std_peaks = s_std.reindex(peaks).mean()
        overall_avg_std = s_std.mean()
        total_std = s_std.sum()
        results.append({
            'Pole': pole,
            'TotalStandardChanges': total_std,
            'PeakCount': len(peaks),
            'AvgOnPeaks': avg_std_peaks,
            'OverallAvg': overall_avg_std
        })
    return pd.DataFrame(results)

# Axe 4 : Analyse unitaire de la volatilité des Cancel Changes
# ------------------------------------------------------------
# Objectif : Analyser la volatilité (écart-type des retours) des Cancel Changes pour chaque entité, sans aucun croisement avec les incidents.
# Méthode :
# - Calcul de la volatilité (rolling std des variations journalières) sur une fenêtre mobile
# Valeur ajoutée : Permet de détecter une instabilité ou une variabilité excessive dans la gestion des changements annulés.
# ------------------------------------------------------------
def compute_cancel_volatility(df_cancel: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Pour chaque entité, calcule la volatilité (écart-type des retours) des Cancel Changes sur une fenêtre mobile.
    """
    results = []
    for pole in df_cancel['Pole'].unique():
        s_cancel = df_cancel[df_cancel['Pole']==pole].set_index('Day')['Value'].sort_index()
        vol = s_cancel.pct_change().rolling(window).std().iloc[-1]
        results.append({'Pole': pole, 'CancelVolatility': vol})
    return pd.DataFrame(results)

# Axe 5 : Analyse unitaire des Fail Changes
# ------------------------------------------------------------
# Objectif : Analyser la volumétrie, la dynamique et les pics de Fail Changes pour chaque entité, sans aucun croisement avec les incidents.
# Méthode :
# - Calcul du nombre total de Fail Changes
# - Détection de pics (jours avec valeur > moyenne + 2*écart-type)
# - Calcul de la moyenne sur les pics et de la moyenne globale
# Valeur ajoutée : Permet de suivre l'activité d'échec de changements et d'alerter sur des périodes d'instabilité ou de gestion défaillante du change.
# ------------------------------------------------------------
def compute_fail_change_stats(df_fail: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Pour chaque entité, calcule le nombre total de Fail Changes, le nombre de pics (jours > moyenne + z_thresh*std), la moyenne sur les pics et la moyenne globale.
    """
    results = []
    for pole in df_fail['Pole'].unique():
        s_fail = df_fail[df_fail['Pole']==pole].set_index('Day')['Value'].sort_index()
        mu, sigma = s_fail.mean(), s_fail.std()
        peaks = s_fail[s_fail > mu + z_thresh * sigma].index
        avg_fail_peaks = s_fail.reindex(peaks).mean()
        overall_avg_fail = s_fail.mean()
        total_fail = s_fail.sum()
        results.append({
            'Pole': pole,
            'TotalFailChanges': total_fail,
            'PeakCount': len(peaks),
            'AvgOnPeaks': avg_fail_peaks,
            'OverallAvg': overall_avg_fail
        })
    return pd.DataFrame(results)

# Axe 6: cycles et rétroactions (incidents → changes)
# Axe 6 : Cycles et rétroactions incidents → changes (lead-lag)
# ------------------------------------------------------------
# Objectif : Analyser si la survenue d'incidents précède une augmentation des changes
# (effet de rétroaction ou de correction post-incident).
# Méthode :
# - Calcul du lead-lag optimal entre incidents et changes
# - Recherche de patterns où incidents précèdent les changes
# Valeur ajoutée : Permet d'anticiper les actions correctives et d'améliorer la résilience.
# ------------------------------------------------------------
def compute_feedback_lead_lag(df_inc: pd.DataFrame, df_chg: pd.DataFrame, max_lag: int = 7) -> pd.DataFrame:
    """
    Analyse lead-lag où les incidents précèdent les changes pour identifier feedback.
    """
    return compute_lead_lag(df_inc, df_chg, max_lag)

# Axe 7: similarité de patterns saisonniers/journaliers
# Axe 7 : Similarité de patterns saisonniers/journaliers
# ------------------------------------------------------------
# Objectif : Mesurer la ressemblance des patterns temporels (saisonnalité, jours de semaine, etc.)
# entre deux séries (ex : incidents vs changes, ou entre deux pôles).
# Méthode :
# - Calcul de la moyenne par jour de semaine et par mois
# - Calcul du coefficient de corrélation entre patterns
# Valeur ajoutée : Détecte des synchronisations ou des ruptures de comportement utiles pour la planification.
# ------------------------------------------------------------
def compute_pattern_similarity(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque entité, calcule la corrélation des moyennes par jour de semaine et par mois.
    """
    results = []
    for pole in set(df1['Pole']).intersection(df2['Pole']):
        s1 = df1[df1['Pole']==pole].set_index('Day')['Value']
        s2 = df2[df2['Pole']==pole].set_index('Day')['Value']
        wk1 = s1.groupby(s1.index.weekday).mean()
        wk2 = s2.groupby(s2.index.weekday).mean()
        wd_corr = wk1.corr(wk2)
        mo1 = s1.groupby(s1.index.month).mean()
        mo2 = s2.groupby(s2.index.month).mean()
        mo_corr = mo1.corr(mo2)
        results.append({'Pole': pole, 'WeekdayCorr': wd_corr, 'MonthCorr': mo_corr})
    return pd.DataFrame(results)

# Axe 8: ratio incidents/changes
def compute_ratio_incidents_changes(df_inc: pd.DataFrame, df_chg: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le ratio volume total d'incidents vs changes par Pole.
    Retourne DataFrame avec colonnes [Pole, IncTotal, ChangeTotal, IncToChangeRatio].
    """
    inc = df_inc.groupby('Pole')['Value'].sum().rename('IncTotal')
    chg = df_chg.groupby('Pole')['Value'].sum().rename('ChangeTotal')
    df = pd.concat([inc, chg], axis=1).fillna(0)
    df['IncToChangeRatio'] = df['IncTotal'] / df['ChangeTotal'].replace(0, np.nan)
    return df.reset_index()

# Axe 9: lead-lag cross-correlation
# Axe 9 : Lead-lag cross-correlation incidents/changes
# ------------------------------------------------------------
# Objectif : Analyser la corrélation temporelle entre incidents et changements pour détecter
# des effets de cause à effet différés (anticipation ou conséquence).
# Méthode :
# - Calcul du coefficient de corrélation pour différents décalages temporels (lags)
# - Identification du lag optimal par entité
# Valeur ajoutée : Permet d’optimiser le timing des actions et d’anticiper les impacts.
# ------------------------------------------------------------
def compute_lead_lag(df1: pd.DataFrame, df2: pd.DataFrame, max_lag: int = 7) -> pd.DataFrame:
    """
    Pour chaque Pole commun, calcule la cross-corr lead-lag et renvoie le lag optimal et le coefficient.
    """
    results = []
    for pole in set(df1['Pole']).intersection(df2['Pole']):
        s1 = df1[df1['Pole']==pole].set_index('Day')['Value'].sort_index()
        s2 = df2[df2['Pole']==pole].set_index('Day')['Value'].sort_index()
        idx = s1.index.union(s2.index)
        s1 = s1.reindex(idx).interpolate().ffill()
        s2 = s2.reindex(idx).interpolate().ffill()
        best_lag, best_corr = 0, -np.inf
        for lag in range(-max_lag, max_lag + 1):
            if lag > 0:
                a, b = s1[:-lag], s2[lag:]
            elif lag < 0:
                a, b = s1[-lag:], s2[:lag]
            else:
                a, b = s1, s2
            if len(a) < 2:
                continue
            corr = a.corr(b)
            if corr > best_corr:
                best_corr, best_lag = corr, lag
        results.append({'Pole': pole, 'BestLag': best_lag, 'BestCorr': best_corr})
    return pd.DataFrame(results)

# Axe 10: cluster d'anomalies
# Axe 10 : Clustering d’anomalies multi-séries
# ------------------------------------------------------------
# Objectif : Détecter les dates où plusieurs séries temporelles présentent simultanément des anomalies,
# afin d'identifier des événements globaux ou systémiques (par exemple, incidents majeurs multi-pôles).
# Méthode :
# - Calcul du z-score pour chaque série et chaque jour
# - Comptage des dates où au moins `min_series` séries sont en anomalie simultanément
# Valeur ajoutée : Permet de détecter des ruptures collectives ou des incidents transverses,
# facilitant l'analyse des causes racines et la priorisation des actions correctives.
# ------------------------------------------------------------
def detect_anomaly_clusters(dfs: Dict[str, pd.DataFrame], z_thresh: float = 2.0, min_series: int = 2) -> List[pd.Timestamp]:
    """
    Identifie les dates où au moins `min_series` séries présentent des anomalies z-score >= z_thresh.
    Retourne liste de dates (pd.Timestamp).
    """
    anomalies = []
    for name, df in dfs.items():
        for pole, g in df.groupby('Pole'):
            series = g.set_index('Day')['Value']
            z = (series - series.mean()) / series.std()
            anomalies.extend(z[abs(z) >= z_thresh].index.tolist())
    counts = pd.Series(anomalies).value_counts()
    return counts[counts >= min_series].index.tolist()

# Axe 11: comparaison incidents internes vs externes
# Axe 11 : Comparaison incidents internes vs externes
# ------------------------------------------------------------
# Objectif : Comparer la volumétrie des incidents internes et externes par entité (P1-P5),
# pour détecter des problématiques d’origine externe ou interne.
# Méthode :
# - Calcul du total d’incidents internes et externes par entité
# - Calcul du ratio incidents externes / internes
# Valeur ajoutée : Permet de cibler les axes d’amélioration (process internes vs dépendances externes).
# ------------------------------------------------------------
def compute_external_internal_ratio(df_int: pd.DataFrame, df_ext: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque niveau P1-P5 le ratio incidents externes / incidents internes.
    Retourne DataFrame avec [Pole, IntTotal, ExtTotal, ExtToIntRatio].
    """
    int_tot = df_int.groupby('Pole')['Value'].sum().rename('IntTotal')
    ext_tot = df_ext.groupby('Pole')['Value'].sum().rename('ExtTotal')
    df = pd.concat([int_tot, ext_tot], axis=1).fillna(0)
    df['ExtToIntRatio'] = df['ExtTotal'] / df['IntTotal'].replace(0, np.nan)
    return df.reset_index()


# Axe 12: MTTR (mean, median, p90, p95)
# Axe 15 : MTTR (Mean Time To Resolve) par pôle et priorité
# ------------------------------------------------------------
# Objectif : Calculer le temps moyen, médian et les percentiles de résolution des incidents
# par entité et priorité (P1–P5).
# Méthode :
# - Calcul de la moyenne, médiane, p90, p95 du temps de résolution
# Valeur ajoutée : Indicateur clé pour le pilotage opérationnel et la détection des zones de fragilité.
# ------------------------------------------------------------
def compute_mttr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule MTTR (mean, median, p90, p95) par Pole et priorité P1–P5.
    df doit contenir ['Pole','Priority','ResolutionTime'] (en heures).
    """
    def q90(x): return x.quantile(0.9)
    def q95(x): return x.quantile(0.95)
    stats = (df.groupby(['Pole','Priority'])['ResolutionTime']
             .agg(['mean','median',q90,q95]))
    stats.columns = ['MTTR_mean','MTTR_median','MTTR_p90','MTTR_p95']
    return stats.reset_index()

# Axe 13: Conformité SLA (% résolus vs breaches)
# Axe 13 : Conformité SLA (% incidents résolus dans les délais)
# ------------------------------------------------------------
# Objectif : Calculer le pourcentage d’incidents résolus dans le délai SLA par priorité (P1/P2),
# et identifier les entités en difficulté.
# Méthode :
# - Calcul du taux de conformité par entité/priorité
# - Détail du nombre d’incidents, résolus à temps, en retard
# Valeur ajoutée : Permet de piloter la performance contractuelle et d’anticiper les pénalités.
# ------------------------------------------------------------
def compute_sla_compliance(df: pd.DataFrame, sla_thresholds: dict) -> pd.DataFrame:
    """
    Calcule % incidents résolus dans SLA vs breach pour P1/P2.
    sla_thresholds ex. {'P1':2,'P2':4} (heures).
    """
    records = []
    for (pole, prio), grp in df.groupby(['Pole','Priority']):
        thresh = sla_thresholds.get(prio)
        if thresh is None:
            continue
        total = len(grp)
        on_time = (grp['ResolutionTime'] <= thresh).sum()
        breach = total - on_time
        records.append({'Pole': pole, 'Priority': prio,
                        'Total': total, 'OnTime': on_time,
                        'Breach': breach,
                        'OnTimePct': on_time/total if total else np.nan})
    return pd.DataFrame(records)

# Axe 14: Statistiques des breaches SLA (max, p90, p95)
# Axe 14 : Statistiques des breaches SLA (max, p90, p95)
# ------------------------------------------------------------
# Objectif : Fournir des statistiques détaillées sur les incidents résolus hors SLA,
# pour chaque entité/priorité.
# Méthode :
# - Calcul du maximum, du 90e et du 95e percentile des temps de résolution hors SLA
# Valeur ajoutée : Permet d’identifier les cas extrêmes et de cibler les processus à améliorer.
# ------------------------------------------------------------
def compute_sla_breach_stats(df: pd.DataFrame, sla_thresholds: dict) -> pd.DataFrame:
    """
    Stats des résolutions hors SLA: max, p90, p95.
    """
    records = []
    for (pole, prio), grp in df.groupby(['Pole','Priority']):
        thresh = sla_thresholds.get(prio)
        if thresh is None:
            continue
        breaches = grp.loc[grp['ResolutionTime'] > thresh, 'ResolutionTime']
        if breaches.empty:
            continue
        records.append({'Pole': pole, 'Priority': prio,
                        'MaxBreach': breaches.max(),
                        'P90Breach': breaches.quantile(0.9),
                        'P95Breach': breaches.quantile(0.95)})
    return pd.DataFrame(records)

# Axe 15: Rolling MTTR (fenêtre glissante de window jours)
# Axe 15 (bis) : MTTR rolling (tendance glissante)
# ------------------------------------------------------------
# Objectif : Suivre l’évolution du MTTR sur une fenêtre glissante pour chaque priorité,
# afin de détecter des tendances ou des ruptures récentes.
# Méthode :
# - Calcul du MTTR moyen sur une fenêtre mobile de `window` jours
# Valeur ajoutée : Permet d’anticiper les dégradations ou les améliorations récentes.
# ------------------------------------------------------------
def compute_mttr_trends(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Rolling MTTR par Priority sur fenêtre de `window` jours.
    df doit contenir ['Day','Priority','ResolutionTime'].
    """
    df2 = df.copy()
    df2['Date'] = pd.to_datetime(df2['Day']).dt.floor('D')
    records = []
    for prio, grp in df2.groupby('Priority'):
        daily = grp.groupby('Date')['ResolutionTime'].mean()
        roll = daily.rolling(window).mean().dropna()
        df_roll = roll.reset_index(name=f'RollingMTTR_{window}d')
        df_roll['Priority'] = prio
        records.append(df_roll)
    return pd.concat(records, ignore_index=True)


def main():
    # --- Chargement des données ---
    # Dans la version finale, tu fourniras directement les DataFrames prêts à l'emploi pour chaque catégorie
    # Exemple :
    #   df_incidents_par_prio = { 'P1': df_p1, 'P2': df_p2, ... }
    #   df_changes_par_type = { 'standard': df_standard, ... }
    #   df_ext, df_mttr, df_breach = ...
    # Le code de chargement CSV et de catégorisation est mis en commentaire ci-dessous :
  
    # === EXEMPLE DE STRUCTURE DE DONNÉES FACTICES POUR TESTER LE PIPELINE ===
    # Tu remplaceras ces DataFrames par tes vraies données quand tu les auras
    # Colonnes minimales attendues : 'Pole', 'day', 'Value' (attention : 'day' en minuscule)
    columns = ['Pole', 'day', 'Value', 'Priority']
    # DataFrames incidents par priorité (P1 à P5)
    df_incidents_par_prio = {prio: pd.DataFrame(columns=columns) for prio in ["P1", "P2", "P3", "P4", "P5"]}
    # DataFrames changes par typologie
    df_changes_par_type = {typ: pd.DataFrame(columns=columns) for typ in ["standard", "emergency", "failed", "cancel", "operated", "total"]}
    # DataFrame incidents externes
    df_ext = pd.DataFrame(columns=columns)
    # DataFrame MTTR
    df_mttr = pd.DataFrame(columns=columns)
    # Découpage MTTR par priorité (pour la première passe)
    df_mttr_p1 = df_mttr[df_mttr['Priority'] == 'P1'].copy()
    df_mttr_p2 = df_mttr[df_mttr['Priority'] == 'P2'].copy()
    # DataFrame SLA breach
    df_breach = pd.DataFrame(columns=columns)
    # --- Fin des exemples factices ---
    # Remplace ces DataFrames par tes vraies données au moment voulu.

    # === ANALYSE PAR AXE (priorité, typologie, etc.) ===
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    syntheses_axe = {}
    prompts_axe = {}

    # === AXES UNITAIRES (PAS DE CROISEMENT INCIDENTS/CHANGES) ===
    # AXE 1 : Analyse unitaire des incidents (P1 à P5)
    for prio in ['P1', 'P2', 'P3', 'P4', 'P5']:
        peak_stats = compute_peak_peaks_analysis(df_incidents_par_prio[prio])
        prompts_axe[f'axe1_peak_corr_{prio}'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de pics et ruptures pour les incidents de priorité {prio} :
{peak_stats.to_string(index=False)}

Votre mission :
- 1. Identifiez les tendances majeures, pics ou ruptures dans la série incidents {prio}, en précisant la date et la valeur pour chaque événement clé.
- 2. Détectez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 recommandations concrètes pour le CTO afin d'améliorer la maîtrise des incidents {prio}.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [AXE1-{prio}], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[AXE1-{prio}] Tendance : ... (date : ..., valeur : ...)
[AXE1-{prio}] Pic détecté : ... (date : ..., valeur : ...)
[AXE1-{prio}] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE1-{prio}] Risque : ...
[AXE1-{prio}] Recommandation : ...
"""
        syntheses_axe[f'axe1_peak_corr_{prio}'] = openai_llm_complete(
            prompt=prompts_axe[f'axe1_peak_corr_{prio}'],
            model=model
        )

    # AXE 2 : Emergency Changes (analyse statistique pure)
    peak_stats_em = compute_change_peaks_analysis(df_changes_par_type['emergency'])
    prompts_axe['axe2_emergency'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de pics et ruptures pour les Emergency Changes :
{peak_stats_em.to_string(index=False)}

Votre mission :
- 1. Identifiez les tendances majeures, pics ou ruptures dans la série Emergency Changes, en précisant la date et la valeur pour chaque événement clé.
- 2. Détectez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 recommandations concrètes pour le CTO afin d'améliorer la maîtrise des Emergency Changes.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [AXE2], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[AXE2] Tendance : ... (date : ..., valeur : ...)
[AXE2] Pic détecté : ... (date : ..., valeur : ...)
[AXE2] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE2] Risque : ...
[AXE2] Recommandation : ...
"""
    syntheses_axe['axe2_emergency'] = openai_llm_complete(
        prompt=prompts_axe['axe2_emergency'],
        model=model
    )

    # AXE 3 : Failed Changes (analyse statistique pure)
    peak_stats_failed = compute_change_peaks_analysis(df_changes_par_type['failed'])
    prompts_axe['axe3_failed'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de pics et ruptures pour les Failed Changes :
{peak_stats_failed.to_string(index=False)}

Votre mission :
- 1. Identifiez les tendances majeures, pics ou ruptures dans la série Failed Changes, en précisant la date et la valeur pour chaque événement clé.
- 2. Détectez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 recommandations concrètes pour le CTO afin d'améliorer la maîtrise des Failed Changes.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [AXE3], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[AXE3] Tendance : ... (date : ..., valeur : ...)
[AXE3] Pic détecté : ... (date : ..., valeur : ...)
[AXE3] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE3] Risque : ...
[AXE3] Recommandation : ...
"""
    syntheses_axe['axe3_failed'] = openai_llm_complete(
        prompt=prompts_axe['axe3_failed'],
        model=model
    )

    # AXE 4 : Cancel Changes (analyse statistique pure)
    peak_stats_cancel = compute_change_peaks_analysis(df_changes_par_type['cancel'])
    prompts_axe['axe4_cancel'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de pics et ruptures pour les Cancel Changes :
{peak_stats_cancel.to_string(index=False)}

Votre mission :
- 1. Identifiez les tendances majeures, pics ou ruptures dans la série Cancel Changes, en précisant la date et la valeur pour chaque événement clé.
- 2. Détectez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 recommandations concrètes pour le CTO afin d'améliorer la maîtrise des Cancel Changes.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [AXE4], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[AXE4] Tendance : ... (date : ..., valeur : ...)
[AXE4] Pic détecté : ... (date : ..., valeur : ...)
[AXE4] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE4] Risque : ...
[AXE4] Recommandation : ...
"""
    syntheses_axe['axe4_cancel'] = openai_llm_complete(
        prompt=prompts_axe['axe4_cancel'],
        model=model
    )

    # AXE 5 : Operated Changes (analyse statistique pure)
    peak_stats_operated = compute_change_peaks_analysis(df_changes_par_type['operated'])
    prompts_axe['axe5_operated'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de pics et ruptures pour les Operated Changes :
{peak_stats_operated.to_string(index=False)}

Votre mission :
- 1. Identifiez les tendances majeures, pics ou ruptures dans la série Operated Changes, en précisant la date et la valeur pour chaque événement clé.
- 2. Détectez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 recommandations concrètes pour le CTO afin d'améliorer la maîtrise des Operated Changes.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [AXE5], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[AXE5] Tendance : ... (date : ..., valeur : ...)
[AXE5] Pic détecté : ... (date : ..., valeur : ...)
[AXE5] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE5] Risque : ...
[AXE5] Recommandation : ...
"""
    syntheses_axe['axe5_operated'] = openai_llm_complete(
        prompt=prompts_axe['axe5_operated'],
        model=model
    )

    # AXE 6 : Standard Changes (analyse statistique pure)
    peak_stats_standard = compute_change_peaks_analysis(df_changes_par_type['standard'])
    prompts_axe['axe6_standard'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de pics et ruptures pour les Standard Changes :
{peak_stats_standard.to_string(index=False)}

Votre mission :
- 1. Identifiez les tendances majeures, pics ou ruptures dans la série Standard Changes, en précisant la date et la valeur pour chaque événement clé.
- 2. Détectez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 recommandations concrètes pour le CTO afin d'améliorer la maîtrise des Standard Changes.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [AXE6], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[AXE6] Tendance : ... (date : ..., valeur : ...)
[AXE6] Pic détecté : ... (date : ..., valeur : ...)
[AXE6] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE6] Risque : ...
[AXE6] Recommandation : ...
"""
    syntheses_axe['axe6_standard'] = openai_llm_complete(
        prompt=prompts_axe['axe6_standard'],
        model=model
    )

#     # Les analyses croisées (corrélations, lead-lag, etc.) sont à traiter uniquement en passe 2 (synthèse globale)
# Votre mission :
# - 1. Analysez la similarité détectée, identifiez toute rupture ou anomalie (date, valeur).
# - 2. Déduisez les risques liés à des patterns saisonniers ou journaliers atypiques.
# - 3. Proposez 2 recommandations pour mieux anticiper les pics saisonniers/journaliers.
# - 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE7], avec date et valeur pour chaque événement.

# Format attendu :
# [AXE7] Tendance : ... (date : ..., valeur : ...)
# [AXE7] Anomalie détectée : ... (date : ..., valeur : ...)
# [AXE7] Risque : ...
# [AXE7] Recommandation : ...
# """
#     syntheses_axe['axe7_pattern_sim'] = openai_llm_complete(
#         prompt=prompts_axe['axe7_pattern_sim'],
#         model=model
#     )

    # AXE 8 : Suivre le ratio global incidents/changes pour détecter une dégradation de la qualité ou un déséquilibre dans le pilotage opérationnel. Indicateur clé pour le management IT.
    inc_chg_ratio = compute_ratio_incidents_changes(pd.concat(df_incidents_par_prio.values(), ignore_index=True), df_changes_par_type['operated'])
    prompts_axe['axe8_inc_chg_ratio'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques du ratio incidents/changes :
{inc_chg_ratio.to_string(index=False)}

Votre mission :
- 1. Analysez la dynamique incidents/changes, détectez toute anomalie, pic ou rupture (date, valeur).
- 2. Identifiez les risques associés à un déséquilibre incidents/changes.
- 3. Proposez 2 recommandations pour optimiser ce ratio.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE8], avec date et valeur pour chaque événement.

Format attendu :
[AXE8] Tendance : ... (date : ..., valeur : ...)
[AXE8] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE8] Risque : ...
[AXE8] Recommandation : ...
"""
    syntheses_axe['axe8_inc_chg_ratio'] = openai_llm_complete(
        prompt=prompts_axe['axe8_inc_chg_ratio'],
        model=model
    )

    # AXE 9 : Analyser la cross-correlation temporelle entre incidents et changements pour détecter des effets de cause à effet différés. Permet d’anticiper les impacts et d’optimiser le timing des actions.
    lead_lag = compute_lead_lag(df_changes_par_type['standard'], pd.concat([df_incidents_par_prio['P1'], df_incidents_par_prio['P2']], ignore_index=True))
    prompts_axe['axe9_lead_lag'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques de lead-lag cross-correlation :
{lead_lag.to_string(index=False)}

Votre mission :
- 1. Analysez la cross-correlation, détectez toute tendance, rupture ou anomalie (date, valeur).
- 2. Identifiez les risques liés à des délais de propagation incidents/changes.
- 3. Proposez 2 recommandations pour réduire ces délais ou en tirer profit.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE9], avec date et valeur pour chaque événement.

Format attendu :
[AXE9] Tendance : ... (date : ..., valeur : ...)
[AXE9] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE9] Risque : ...
[AXE9] Recommandation : ...
"""
    syntheses_axe['axe9_lead_lag'] = openai_llm_complete(
        prompt=prompts_axe['axe9_lead_lag'],
        model=model
    )

    # AXE 10 : Identifier les clusters d'anomalies multi-séries pour détecter des crises ou des événements majeurs. Permet une réaction rapide et une priorisation des investigations.
    # On ne conserve que les priorités critiques P1 et P2 pour limiter la volumétrie
    dfs_anomaly = {k: v for k, v in df_incidents_par_prio.items() if k in ['P1', 'P2']}
    anomaly_dates = detect_anomaly_clusters(dfs_anomaly)
    prompts_axe['axe10_anomaly_clusters'] = f"""
Vous êtes un expert IT analyste. Voici les dates de clusters d'anomalies détectés sur plusieurs séries :
{list(anomaly_dates)}

Votre mission :
- 1. Analysez les clusters détectés, identifiez leur signification, leur gravité et leur impact (date, valeur).
- 2. Déduisez les risques majeurs associés à ces clusters.
- 3. Proposez 2 recommandations pour anticiper ou réagir à ces clusters.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE10], avec date et valeur pour chaque événement.

Format attendu :
[AXE10] Cluster détecté : ... (date : ..., valeur : ...)
[AXE10] Risque : ...
[AXE10] Recommandation : ...
"""
    syntheses_axe['axe10_anomaly_clusters'] = openai_llm_complete(
        prompt=prompts_axe['axe10_anomaly_clusters'],
        model=model
    )

    # AXE 11 : Suivre la part des incidents externes vs internes pour piloter la relation fournisseurs et la maîtrise du SI. Sert à cibler les actions correctives et négociations externes.
    # On ne conserve que les incidents P1 et P2 pour la comparaison externe/interne
    df_inc_p1p2 = pd.concat([v for k, v in df_incidents_par_prio.items() if k in ['P1', 'P2']], ignore_index=True)
    ext_int_ratio = compute_external_internal_ratio(df_inc_p1p2, df_ext)
    prompts_axe['axe11_ext_int_ratio'] = f"""
Vous êtes un expert IT analyste. Voici les statistiques du ratio incidents externes / incidents internes :
{ext_int_ratio.to_string(index=False)}

Votre mission :
- 1. Analysez ce ratio, détectez toute anomalie ou rupture (date, valeur).
- 2. Identifiez les risques associés à une hausse des incidents externes par rapport aux internes.
- 3. Proposez 2 recommandations pour rééquilibrer ce ratio.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE11], avec date et valeur pour chaque événement.

Format attendu :
[AXE11] Tendance : ... (date : ..., valeur : ...)
[AXE11] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE11] Risque : ...
[AXE11] Recommandation : ...
"""
    syntheses_axe['axe11_ext_int_ratio'] = openai_llm_complete(
        prompt=prompts_axe['axe11_ext_int_ratio'],
        model=model
    )

    # === AXES 12 à 15 : Calculs et prompts à partir des fonctions métiers ===
    # 1. Calculs des valeurs MTTR et SLA
    mttr_p1 = compute_mttr(df_mttr[df_mttr['Priority'] == 'P1'])
    mttr_p2 = compute_mttr(df_mttr[df_mttr['Priority'] == 'P2'])
    mttr_p1_sla_breach = compute_mttr(df_mttr[(df_mttr['Priority'] == 'P1') & (df_mttr['SLA_Breach'] == True)])
    mttr_p2_sla_breach = compute_mttr(df_mttr[(df_mttr['Priority'] == 'P2') & (df_mttr['SLA_Breach'] == True)])

    # 2. Génération des prompts directement avec les variables
    prompts_axe['axe12_mttr_p1'] = f"""
Vous êtes un expert IT analyste. MTTR pour la priorité P1 :
MTTR_P1 = {mttr_p1}

Votre mission :
- 1. Analysez la tendance du MTTR P1, détectez toute dérive ou anomalie (date, valeur).
- 2. Proposez 2 recommandations pour améliorer le MTTR sur cette priorité.
- 3. Formatez la synthèse en bullet points, chaque point commençant par [AXE12_P1], avec date et valeur pour chaque événement.

Format attendu :
[AXE12_P1] Tendance : ... (date : ..., valeur : ...)
[AXE12_P1] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE12_P1] Recommandation : ...
"""
    syntheses_axe['axe12_mttr_p1'] = openai_llm_complete(
        prompt=prompts_axe['axe12_mttr_p1'],
        model=model
    )

    prompts_axe['axe12_mttr_p2'] = f"""
Vous êtes un expert IT analyste. MTTR pour la priorité P2 :
MTTR_P2 = {mttr_p2}

Votre mission :
- 1. Analysez la tendance du MTTR P2, détectez toute dérive ou anomalie (date, valeur).
- 2. Proposez 2 recommandations pour améliorer le MTTR sur cette priorité.
- 3. Formatez la synthèse en bullet points, chaque point commençant par [AXE12_P2], avec date et valeur pour chaque événement.

Format attendu :
[AXE12_P2] Tendance : ... (date : ..., valeur : ...)
[AXE12_P2] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE12_P2] Recommandation : ...
"""
    syntheses_axe['axe12_mttr_p2'] = openai_llm_complete(
        prompt=prompts_axe['axe12_mttr_p2'],
        model=model
    )

    prompts_axe['axe13_sla_compliance'] = f"""
Vous êtes un expert IT analyste. MTTR et conformité SLA pour P1 et P2 :
MTTR_P1 = {mttr_p1}
MTTR_P2 = {mttr_p2}
MTTR_P1_SLA_BREACH = {mttr_p1_sla_breach}
MTTR_P2_SLA_BREACH = {mttr_p2_sla_breach}

Votre mission :
- 1. Analysez le respect des SLA, détectez toute dérive ou anomalie (date, valeur).
- 2. Identifiez les risques majeurs liés aux non-conformités.
- 3. Proposez 2 recommandations pour améliorer la conformité SLA.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE13], avec date et valeur pour chaque événement.

Format attendu :
[AXE13] Tendance : ... (date : ..., valeur : ...)
[AXE13] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE13] Risque : ...
[AXE13] Recommandation : ...
"""
    syntheses_axe['axe13_sla_compliance'] = openai_llm_complete(
        prompt=prompts_axe['axe13_sla_compliance'],
        model=model
    )

    for prio in ['P1', 'P2']:
        breach_key = f"mttr_{prio.lower()}_sla_breach"
        prompts_axe[f'axe14_sla_breach_{prio}'] = f"""
Vous êtes un expert IT analyste. MTTR des incidents hors SLA pour la priorité {prio} :
MTTR_{prio}_SLA_BREACH = {mttr_p1_sla_breach if prio == 'P1' else mttr_p2_sla_breach}

Votre mission :
- 1. Analysez les cas de breach SLA, détectez toute anomalie ou tendance (date, valeur).
- 2. Identifiez les risques majeurs liés aux breaches.
- 3. Proposez 2 recommandations pour réduire les breaches SLA.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE14-{prio}], avec date et valeur pour chaque événement.

Format attendu :
[AXE14-{prio}] Tendance : ... (date : ..., valeur : ...)
[AXE14-{prio}] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE14-{prio}] Risque : ...
[AXE14-{prio}] Recommandation : ...
"""
        syntheses_axe[f'axe14_sla_breach_{prio}'] = openai_llm_complete(
            prompt=prompts_axe[f'axe14_sla_breach_{prio}'],
            model=model
        )

    prompts_axe['axe15_mttr_trends'] = f"""
Vous êtes un expert IT analyste. MTTR glissant (rolling) pour P1 et P2 :
MTTR_P1 = {mttr_p1}
MTTR_P2 = {mttr_p2}
MTTR_P1_SLA_BREACH = {mttr_p1_sla_breach}
MTTR_P2_SLA_BREACH = {mttr_p2_sla_breach}

Votre mission :
- 1. Analysez les tendances du MTTR glissant, détectez toute anomalie ou rupture (date, valeur).
- 2. Identifiez les risques liés à une dégradation du MTTR sur la période.
- 3. Proposez 2 recommandations pour améliorer le MTTR glissant.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [AXE15], avec date et valeur pour chaque événement.

Format attendu :
[AXE15] Tendance : ... (date : ..., valeur : ...)
[AXE15] Anomalie détectée : ... (date : ..., valeur : ...)
[AXE15] Risque : ...
[AXE15] Recommandation : ...
"""
    syntheses_axe['axe15_mttr_trends'] = openai_llm_complete(
        prompt=prompts_axe['axe15_mttr_trends'],
        model=model
    )

    # === AXES GLOBAUX ===
    # Synthèse globale multi-axes (incidents, changes, MTTR, SLA, etc.)
    syntheses_concat = "\n".join([
        f"Synthèse {k} :\n{v}" for k, v in list(syntheses_axe.items()) if k.startswith('axe') and not k.startswith('axe_global')
    ])
    prompts_axe['axe_global_synthese'] = f"""
Vous êtes un expert IT senior. Voici la synthèse des analyses détaillées sur 15 axes (incidents, changes, MTTR, SLA, corrélations, patterns, etc.) :

{syntheses_concat}

Votre mission :
1. Analysez l’ensemble des synthèses pour détecter :
   - Des corrélations, causalités ou ruptures entre incidents et changes (tous types confondus).
   - Des patterns saisonniers ou journaliers communs ou divergents entre incidents et changes.
   - Des liens entre la dynamique des MTTR, la conformité SLA et l’évolution des incidents/changes.
   - Des clusters d’anomalies multi-axes ou des signaux faibles transverses.
2. Identifiez les risques systémiques ou globaux, et les axes de fragilité du SI.
3. Proposez des plans d’action concrets et priorisés à destination du CTO pour améliorer la maîtrise opérationnelle.
4. Listez les quick-wins et les actions structurantes, en expliquant pour chaque action à quel(s) axe(s) elle se rattache.
5. Reformulez la synthèse globale en bullet points, chaque point commençant par [GLOBAL], et précisez systématiquement le ou les axes concernés.

Format attendu :
[GLOBAL][AXE1, AXE5] Anomalie transversale détectée : ...
[GLOBAL][AXE2, AXE12] Risque global : ...
[GLOBAL][AXE7, AXE10] Plan d’action prioritaire : ...
[GLOBAL][AXE3, AXE14] Quick-win : ...
[GLOBAL][AXE...] Autre point clé : ...
"""
    syntheses_axe['axe_global_synthese'] = openai_llm_complete(
        prompt=prompts_axe['axe_global_synthese'],
        model=model
    )

    print("\n=== SYNTHESE GLOBALE LLM ===\n" + syntheses_axe.get('axe_global_synthese', 'Aucune synthèse globale générée.'))

if __name__ == '__main__':
    syntheses_axe = main()
    print("\n=== SYNTHESE PAR AXE ===")
    for axe, synth in syntheses_axe.items():
        if axe != 'axe_global_synthese':
            print(f"\n--- {axe.upper()} ---\n{synth}")
    print("\n=== SYNTHESE GLOBALE LLM ===\n" + syntheses_axe.get('axe_global_synthese', 'Aucune synthèse globale générée.'))
