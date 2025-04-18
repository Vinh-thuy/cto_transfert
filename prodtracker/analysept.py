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


def compute_key_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute optimized set of metrics by family:
      - Volatility: std and Bollinger %B on daily values
      - Momentum: slope of MA07 and ADX on MA07
      - Risk: max drawdown on cumulative daily
      - Seasonality: mean month-over-month growth on cumulative monthly
    """
    df = df.sort_values(['Pole', 'Day']).copy()
    results = []
    for pole, g in df.groupby('Pole'):
        # Volatility on raw daily values
        vals = g['Value']
        vol_std = vals.std()
        m = vals.rolling(20).mean()
        s = vals.rolling(20).std()
        boll_pctB = ((vals.iloc[-1] - m.iloc[-1]) / (2 * s.iloc[-1])) if s.iloc[-1] else np.nan

        # Momentum on MA07
        ma07 = g['MA07']
        days = np.arange(len(ma07))
        slope = np.gradient(ma07, days).mean()
        # ADX approximation on MA07
        diff = np.diff(ma07.replace(np.nan, method='ffill'))
        plus_dm = np.maximum(diff, 0)
        minus_dm = np.maximum(-diff, 0)
        tr = np.abs(diff)
        atr = pd.Series(tr).rolling(14).mean().iloc[-1]
        plus_di = 100 * pd.Series(plus_dm).rolling(14).mean().iloc[-1] / atr if atr else np.nan
        minus_di = 100 * pd.Series(minus_dm).rolling(14).mean().iloc[-1] / atr if atr else np.nan
        adx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di) if (plus_di + minus_di) else np.nan

        # Risk: max drawdown on cumulative daily
        cum = g['CumDaily']
        peak = cum.cummax()
        dd = (cum - peak) / peak
        max_dd = dd.min()

        # Seasonality: mean MoM on cumulative monthly
        mom = g['CumMonthly'].pct_change().dropna()
        mom_mean = mom.mean()

        results.append({
            'Pole': pole,
            'Vol_Std': vol_std,
            'BollPctB': boll_pctB,
            'Slope_MA07': slope,
            'ADX_MA07': adx,
            'MaxDrawdown': max_dd,
            'MoM_Mean': mom_mean
        })
    return pd.DataFrame(results)


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


# Axe 1: corrélation pics de changes vs incidents (P1/P2)
def compute_peak_correlation(df_inc: pd.DataFrame, df_chg: pd.DataFrame, z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Pour chaque entité (Pole), identifie les jours où les changes dépassent mean+z_thresh*std,
    calcule le nombre de pics, la moyenne d'incidents (P1/P2) ces jours-là, la moyenne globale et la corrélation.
    """
    results = []
    # filtrer incidents P1/P2
    df_inc_crit = df_inc[df_inc['Pole'].isin(['P1','P2'])]
    for pole in set(df_chg['Pole']).intersection(df_inc_crit['Pole']):
        s_chg = df_chg[df_chg['Pole']==pole].set_index('Day')['Value'].sort_index()
        mu, sigma = s_chg.mean(), s_chg.std()
        peaks = s_chg[s_chg > mu + z_thresh * sigma].index
        s_inc = df_inc_crit[df_inc_crit['Pole']==pole].set_index('Day')['Value'].sort_index()
        avg_inc_peaks = s_inc.reindex(peaks).mean()
        overall_avg_inc = s_inc.mean()
        corr = s_chg.corr(s_inc)
        results.append({'Pole': pole, 'PeakCount': len(peaks), 'AvgIncOnPeaks': avg_inc_peaks,
                        'OverallAvgInc': overall_avg_inc, 'Correlation': corr})
    return pd.DataFrame(results)

# Axe 2: impact des Emergency Changes sur incidents critiques
def compute_emergency_impact(df_em: pd.DataFrame, df_inc: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque entité, calcule la fréquence de Emergency Changes (jours avec value>0) et
    la corrélation avec incidents critiques (P1/P2).
    """
    results = []
    df_inc_crit = df_inc[df_inc['Pole'].isin(['P1','P2'])]
    for pole in set(df_em['Pole']).intersection(df_inc_crit['Pole']):
        s_em = df_em[df_em['Pole']==pole].set_index('Day')['Value'].sort_index()
        s_inc = df_inc_crit[df_inc_crit['Pole']==pole].set_index('Day')['Value'].sort_index()
        freq = (s_em > 0).sum()
        corr = s_em.corr(s_inc)
        results.append({'Pole': pole, 'EmergencyFreq': freq, 'CorrEmInc': corr})
    return pd.DataFrame(results)

# Axe 3: ratio Standard Changes vs incidents mineurs (P3-P5)
def compute_standard_minor_ratio(df_std: pd.DataFrame, df_inc: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule pour chaque entité le ratio total Standard Changes / incidents mineurs (P3,P4,P5).
    """
    results = []
    df_inc_min = df_inc[df_inc['Pole'].isin(['P3','P4','P5'])]
    for pole in set(df_std['Pole']).intersection(df_inc_min['Pole']):
        total_std = df_std[df_std['Pole']==pole]['Value'].sum()
        total_min = df_inc_min[df_inc_min['Pole']==pole]['Value'].sum()
        ratio = total_std / total_min if total_min else np.nan
        results.append({'Pole': pole, 'StdToMinorIncRatio': ratio})
    return pd.DataFrame(results)

# Axe 4: effet des Cancel Changes sur la volatilité des incidents
def compute_cancel_vs_incident_volatility(df_cancel: pd.DataFrame, df_inc: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Pour chaque entité, calcule la volatilité (std des retours) des incidents et la corrélation
    avec les Cancel Changes.
    """
    results = []
    for pole in set(df_cancel['Pole']).intersection(df_inc['Pole']):
        s_cancel = df_cancel[df_cancel['Pole']==pole].set_index('Day')['Value'].sort_index()
        s_inc = df_inc[df_inc['Pole']==pole].set_index('Day')['Value'].sort_index()
        vol = s_inc.pct_change().rolling(window).std().iloc[-1]
        corr = s_cancel.corr(s_inc)
        results.append({'Pole': pole, 'IncVolatility': vol, 'CancelIncCorr': corr})
    return pd.DataFrame(results)

# Axe 5: corrélation Fail Changes et incidents critiques
def compute_fail_incident_correlation(df_fail: pd.DataFrame, df_inc: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque entité, calcule la corrélation entre Fail Changes et incidents critiques (P1/P2).
    """
    results = []
    df_inc_crit = df_inc[df_inc['Pole'].isin(['P1','P2'])]
    for pole in set(df_fail['Pole']).intersection(df_inc_crit['Pole']):
        s_fail = df_fail[df_fail['Pole']==pole].set_index('Day')['Value'].sort_index()
        s_inc = df_inc_crit[df_inc_crit['Pole']==pole].set_index('Day')['Value'].sort_index()
        corr = s_fail.corr(s_inc)
        results.append({'Pole': pole, 'FailIncCorr': corr})
    return pd.DataFrame(results)

# Axe 6: cycles et rétroactions (incidents → changes)
def compute_feedback_lead_lag(df_inc: pd.DataFrame, df_chg: pd.DataFrame, max_lag: int = 7) -> pd.DataFrame:
    """
    Analyse lead-lag où les incidents précèdent les changes pour identifier feedback.
    """
    return compute_lead_lag(df_inc, df_chg, max_lag)

# Axe 7: similarité de patterns saisonniers/journaliers
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
def compute_ratio_incidents_changes(df_inc: pd.DataFrame, df_change: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule le ratio volume total d'incidents vs changes par Pole.
    Retourne DataFrame avec colonnes [Pole, IncTotal, ChangeTotal, IncToChangeRatio].
    """
    inc = df_inc.groupby('Pole')['Value'].sum().rename('IncTotal')
    chg = df_change.groupby('Pole')['Value'].sum().rename('ChangeTotal')
    df = pd.concat([inc, chg], axis=1).fillna(0)
    df['IncToChangeRatio'] = df['IncTotal'] / df['ChangeTotal'].replace(0, np.nan)
    return df.reset_index()

# Axe 9: lead-lag cross-correlation
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
    columns = ['Pole', 'day', 'Value']
    # DataFrames incidents par priorité (P1 à P5)
    df_incidents_par_prio = {prio: pd.DataFrame(columns=columns) for prio in ["P1", "P2", "P3", "P4", "P5"]}
    # DataFrames changes par typologie
    df_changes_par_type = {typ: pd.DataFrame(columns=columns) for typ in ["standard", "emergency", "failed", "cancel", "operated", "total"]}
    # DataFrame incidents externes
    df_ext = pd.DataFrame(columns=columns)
    # DataFrame MTTR
    df_mttr = pd.DataFrame(columns=columns)
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

    # Incidents internes par priorité
    for prio, df in df_incidents_par_prio.items():
        axe = f"incidents_P{prio}"
        prompts_axe[axe] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents internes priorité {prio} :\n{df.head(10).to_string(index=False)}\n
Votre mission :
- 1. Dégagez les tendances majeures (volumes, évolutions, pics), précisez pour chaque événement clé la date et la valeur.
- 2. Identifiez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 à 3 recommandations concrètes pour le CTO.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [INCIDENT_P{prio}], et pour chaque événement, indiquez toujours la date et la valeur.
"""
        try:
            syntheses_axe[axe] = openai_llm_complete(
                prompt=prompts_axe[axe],
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            print(f"[LLM][{axe}] Réponse OK.")
        except Exception as e:
            print(f"[LLM][{axe}] Erreur : {e}")
            syntheses_axe[axe] = f"[ERREUR LLM] {e}"

    # Changes par typologie
    for typ, df in df_changes_par_type.items():
        axe = f"changes_{typ}"
        prompts_axe[axe] = f"""
Vous êtes un expert IT analyste. Voici les données de changes typologie {typ} :\n{df.head(10).to_string(index=False)}\n
Votre mission :
- 1. Dégagez les tendances majeures (volumes, évolutions, pics), précisez pour chaque événement clé la date et la valeur.
- 2. Identifiez les points critiques, échecs, annulations, urgences, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 à 3 recommandations concrètes pour le CTO.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [CHANGE_{typ.upper()}], et pour chaque événement, indiquez toujours la date et la valeur.
"""
        try:
            syntheses_axe[axe] = openai_llm_complete(
                prompt=prompts_axe[axe],
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            print(f"[LLM][{axe}] Réponse OK.")
        except Exception as e:
            print(f"[LLM][{axe}] Erreur : {e}")
            syntheses_axe[axe] = f"[ERREUR LLM] {e}"

    # Incidents externes
    if df_ext is not None:
        axe = "incidents_externes"
        prompts_axe[axe] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents externes :\n{df_ext.head(10).to_string(index=False)}\n
Votre mission :
- 1. Analysez les tendances et impacts sur la qualité de service, précisez pour chaque événement clé la date et la valeur.
- 2. Détectez toute anomalie ou risque fournisseur, et indiquez la date et la valeur.
- 3. Proposez 2 recommandations pour le CTO.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [EXTERNE], et pour chaque événement, indiquez toujours la date et la valeur.
"""
        try:
            syntheses_axe[axe] = openai_llm_complete(
                prompt=prompts_axe[axe],
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            print(f"[LLM][{axe}] Réponse OK.")
        except Exception as e:
            print(f"[LLM][{axe}] Erreur : {e}")
            syntheses_axe[axe] = f"[ERREUR LLM] {e}"

    # MTTR
    if df_mttr is not None:
        axe = "mttr"
        prompts_axe[axe] = f"""
Vous êtes un expert IT analyste. Voici les données de temps moyen de résolution (MTTR) :\n{df_mttr.head(10).to_string(index=False)}\n
Votre mission :
- 1. Analysez la performance globale (niveaux, variations, dérives), précisez pour chaque dérive ou anomalie la date et la valeur.
- 2. Proposez 2 recommandations pour améliorer le MTTR.
- 3. Formatez la synthèse en bullet points, chaque point commençant par [MTTR], et pour chaque événement, indiquez toujours la date et la valeur.
"""
        try:
            syntheses_axe[axe] = openai_llm_complete(
                prompt=prompts_axe[axe],
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            print(f"[LLM][{axe}] Réponse OK.")
        except Exception as e:
            print(f"[LLM][{axe}] Erreur : {e}")
            syntheses_axe[axe] = f"[ERREUR LLM] {e}"

    # SLA breach
    if df_breach is not None:
        axe = "sla_breach"
        prompts_axe[axe] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents hors SLA :\n{df_breach.head(10).to_string(index=False)}\n
Votre mission :
- 1. Identifiez les causes principales des non-conformités, précisez pour chaque non-conformité la date et la valeur.
- 2. Évaluez le risque global pour le service.
- 3. Proposez 2 mesures correctives prioritaires.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [SLA], et pour chaque événement, indiquez toujours la date et la valeur.
"""
        try:
            syntheses_axe[axe] = openai_llm_complete(
                prompt=prompts_axe[axe],
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            print(f"[LLM][{axe}] Réponse OK.")
        except Exception as e:
            print(f"[LLM][{axe}] Erreur : {e}")
            syntheses_axe[axe] = f"[ERREUR LLM] {e}"

    # Affichage des synthèses par axe
    print("\n=== SYNTHESES LLM PAR AXE ===")
    for axe, synth in syntheses_axe.items():
        print(f"\n[AXE: {axe}]\n{synth}\n")

    # Génération du prompt global et appel LLM pour la synthèse globale
    prompt_global = generate_global_prompt({}, syntheses_axe)
    print("\n=== PROMPT GLOBAL CTO ===\n" + prompt_global)
    print("\n=== APPEL LLM GLOBAL ===")
    try:
        synthese_globale = openai_llm_complete(
            prompt=prompt_global,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        print("\n=== SYNTHESE GLOBALE LLM ===\n" + synthese_globale)
    except Exception as e:
        print(f"[LLM][GLOBAL] Erreur : {e}")
        synthese_globale = f"[ERREUR LLM GLOBAL] {e}"


    # Analyse intermédiaire LLM par axe d'analyse
    syntheses_axe = {}
    prompts_axe = {}
    # 1. Incidents internes
    df_inc = pd.concat(df_incidents_par_prio.values(), ignore_index=True)
    prompts_axe['incidents'] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents internes (toutes priorités et périodes confondues) :\n{df_inc.head(10).to_string(index=False)}\n
Votre mission :
- 1. Dégagez les tendances majeures (volumes, évolutions, pics) même à partir de simples comptages, et pour chaque événement clé détecté, précisez la date et la valeur associée.
- 2. Identifiez toute anomalie, rupture ou signal faible, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 à 3 recommandations concrètes pour le CTO.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [INCIDENT], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[INCIDENT] Tendance majeure : ... (date : ..., valeur : ...)
[INCIDENT] Pic détecté : ... (date : ..., valeur : ...)
[INCIDENT] Anomalie détectée : ... (date : ..., valeur : ...)
[INCIDENT] Risque : ...
[INCIDENT] Recommandation : ...
"""
    # 2. Changes
    df_chg = pd.concat(df_changes_par_type.values(), ignore_index=True)
    prompts_axe['changes'] = f"""
Vous êtes un expert IT analyste. Voici les données de changes (toutes typologies et périodes confondues) :\n{df_chg.head(10).to_string(index=False)}\n
Votre mission :
- 1. Dégagez les tendances majeures (volumes, évolutions, pics) même à partir de simples comptages, et pour chaque point critique, échec, annulation ou urgence détecté, précisez la date et la valeur associée.
- 2. Identifiez les points critiques, échecs, annulations, urgences, et indiquez la date et la valeur.
- 3. Listez les risques ou alertes à surveiller.
- 4. Proposez 2 à 3 recommandations concrètes pour le CTO.
- 5. Formatez la synthèse en bullet points, chaque point commençant par [CHANGE], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[CHANGE] Tendance majeure : ... (date : ..., valeur : ...)
[CHANGE] Point critique : ... (date : ..., valeur : ...)
[CHANGE] Risque : ...
[CHANGE] Recommandation : ...
"""
    # 3. MTTR
    if df_mttr is not None:
        prompts_axe['mttr'] = f"""
Vous êtes un expert IT analyste. Voici les données de temps moyen de résolution (MTTR) :\n{df_mttr.head(10).to_string(index=False)}\n
Votre mission :
- 1. Analysez la performance globale (niveaux, variations, dérives), même à partir de simples moyennes, et pour chaque dérive ou anomalie détectée, précisez la date et la valeur.
- 2. Détectez toute anomalie ou rupture, et indiquez la date et la valeur.
- 3. Proposez 2 recommandations pour améliorer le MTTR.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [MTTR], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[MTTR] Performance : ... (date : ..., valeur : ...)
[MTTR] Anomalie : ... (date : ..., valeur : ...)
[MTTR] Recommandation : ...
"""
    # 4. SLA breach
    if df_breach is not None:
        prompts_axe['sla'] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents hors SLA :\n{df_breach.head(10).to_string(index=False)}\n
Votre mission :
- 1. Identifiez les causes principales des non-conformités, et pour chaque non-conformité détectée, précisez la date et la valeur.
- 2. Évaluez le risque global pour le service.
- 3. Proposez 2 mesures correctives prioritaires.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [SLA], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[SLA] Cause : ... (date : ..., valeur : ...)
[SLA] Risque : ...
[SLA] Mesure corrective : ...
"""
    # 5. Incidents externes
    if df_ext is not None:
        prompts_axe['externes'] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents externes :\n{df_ext.head(10).to_string(index=False)}\n
Votre mission :
- 1. Analysez les tendances et impacts sur la qualité de service, même à partir de simples comptages, et pour chaque événement clé, précisez la date et la valeur.
- 2. Détectez toute anomalie ou risque fournisseur, et indiquez la date et la valeur.
- 3. Proposez 2 recommandations pour le CTO.
- 4. Formatez la synthèse en bullet points, chaque point commençant par [EXTERNE], et pour chaque événement, indiquez toujours la date et la valeur.

Format attendu :
[EXTERNE] Tendance : ... (date : ..., valeur : ...)
[EXTERNE] Anomalie : ... (date : ..., valeur : ...)
[EXTERNE] Recommandation : ...
"""

    # Appel LLM pour chaque axe d'analyse
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

    print("\n=== APPEL LLM PAR AXE ===")
    for axe, prompt in prompts_axe.items():
        print(f"[LLM][{axe}] Appel en cours...")
        try:
            syntheses_axe[axe] = openai_llm_complete(
                prompt=prompt,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            print(f"[LLM][{axe}] Réponse OK.")
        except Exception as e:
            print(f"[LLM][{axe}] Erreur : {e}")
            syntheses_axe[axe] = f"[ERREUR LLM] {e}"

    # Génération du prompt CTO (prompt système)
    # prompt_global = generate_global_prompt(all_metrics, syntheses_axe)
    prompt_global = generate_global_prompt(syntheses_axe)
    print("\n=== PROMPT GLOBAL CTO ===\n" + prompt_global)

    # Appel LLM pour la synthèse globale
    print("\n=== APPEL LLM GLOBAL ===")
    try:
        synthese_globale = openai_llm_complete(
            prompt=prompt_global,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        print("\n=== SYNTHESE GLOBALE LLM ===\n" + synthese_globale)
    except Exception as e:
        print(f"[LLM][GLOBAL] Erreur : {e}")
        synthese_globale = f"[ERREUR LLM GLOBAL] {e}"
    # prompt_cto = generate_global_prompt(all_metrics)
    # print("\n=== PROMPT CTO POUR SYSTEME LLM/IA ===\n")
    # print(prompt_cto)

if __name__ == '__main__':
    main()
