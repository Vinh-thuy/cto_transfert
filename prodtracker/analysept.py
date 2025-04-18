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
    import sys
    from pathlib import Path
    files = sys.argv[1:]
    if not files:
        print("Usage: python analysis_metrics_optimized.py <csv1> [<csv2> ...]")
        return
    all_metrics = {}
    raw_dfs = {}
    # Catégorisation explicite des datasets
    incident_names = []
    change_names = []
    mttr_names = []
    breach_sla_names = []
    external_names = []
    for path in files:
        df = pd.read_csv(path, parse_dates=['Day'])
        if df.columns[0] != 'Pole':
            df = df.rename(columns={df.columns[0]: 'Pole'})
        name = Path(path).stem
        raw_dfs[name] = df
        lname = name.lower()
        if "external" in lname:
            external_names.append(name)
        elif any(x in lname for x in ["change", "operated", "standard", "cancelled", "failed", "emergency"]):
            change_names.append(name)
        elif any(x in lname for x in ["p1 incident", "p2 incident", "p3 incident", "p4 incident", "p5 incident"]):
            incident_names.append(name)
        elif any(x in lname for x in ["mttr", "resolution_time_avg"]):
            mttr_names.append(name)
        elif any(x in lname for x in ["breach", "breaching_sla"]):
            breach_sla_names.append(name)
        metrics = compute_key_metrics(df)
        metrics_dict = metrics.set_index('Pole').to_dict(orient='index')
        print(f"\n--- Prompt pour {name} ---\n" + generate_prompt(metrics_dict, name))
        all_metrics[name] = metrics_dict
    # Date de référence
    import argparse
    from datetime import datetime
    parser = argparse.ArgumentParser()
    parser.add_argument('--refdate', type=str, help="Date de référence au format YYYY-MM-DD (optionnel)")
    args, _ = parser.parse_known_args()
    if args.refdate:
        ref_date = pd.to_datetime(args.refdate)
    else:
        ref_date = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
    def last_n_days(df, n):
        return df[df['Day'] >= (ref_date - pd.Timedelta(days=n-1))]

    # Analyse incidents internes
    if incident_names:
        df_inc = pd.concat([raw_dfs[n] for n in incident_names], ignore_index=True)
    else:
        df_inc = None
    # Analyse incidents externes
    if external_names:
        df_ext = pd.concat([raw_dfs[n] for n in external_names], ignore_index=True)
    else:
        df_ext = None
    # Analyse changes
    if change_names:
        df_chg = pd.concat([raw_dfs[n] for n in change_names], ignore_index=True)
    else:
        df_chg = None
    # Analyse MTTR
    if mttr_names:
        df_mttr = pd.concat([raw_dfs[n] for n in mttr_names], ignore_index=True)
    else:
        df_mttr = None
    # Analyse SLA breach
    if breach_sla_names:
        df_breach = pd.concat([raw_dfs[n] for n in breach_sla_names], ignore_index=True)
    else:
        df_breach = None

    # Fenêtres multi-période
    for window in [7, 30, 90]:
        # Incidents internes
        if df_inc is not None:
            df_inc_win = last_n_days(df_inc, window)
            all_metrics[f'IncidentCounts_{window}d'] = df_inc_win.groupby('Pole')['Value'].sum().to_dict()
            all_metrics[f'IncidentAnomalies_{window}d'] = detect_anomaly_clusters({'incidents': df_inc_win})
        # Incidents externes
        if df_ext is not None:
            df_ext_win = last_n_days(df_ext, window)
            all_metrics[f'ExternalIncidentCounts_{window}d'] = df_ext_win.groupby('Pole')['Value'].sum().to_dict()
            all_metrics[f'ExternalIncidentAnomalies_{window}d'] = detect_anomaly_clusters({'incidents_ext': df_ext_win})
        # Changes
        if df_chg is not None:
            df_chg_win = last_n_days(df_chg, window)
            for typ in ['standard','emergency','failed','cancel','total','operated']:
                typ_names = [n for n in change_names if typ in n.lower()]
                if typ_names:
                    df_typ = pd.concat([raw_dfs[n] for n in typ_names], ignore_index=True)
                    df_typ_win = last_n_days(df_typ, window)
                    all_metrics[f'Change_{typ.capitalize()}_{window}d'] = df_typ_win.groupby('Pole')['Value'].sum().to_dict()
            all_metrics[f'ChangeAnomalies_{window}d'] = detect_anomaly_clusters({'changes': df_chg_win})
            if df_inc is not None:
                all_metrics[f'IncToChangeRatio_{window}d'] = compute_ratio_incidents_changes(df_inc_win, df_chg_win).set_index('Pole').to_dict(orient='index')
        # MTTR
        if df_mttr is not None:
            df_mttr_win = last_n_days(df_mttr, window)
            all_metrics[f'MTTR_{window}d'] = df_mttr_win.groupby('Pole')['Value'].mean().to_dict()
        # SLA breach : taux
        if df_breach is not None and df_mttr is not None:
            df_breach_win = last_n_days(df_breach, window)
            df_mttr_win = last_n_days(df_mttr, window)
            for pole in df_breach_win['Pole'].unique():
                total = df_mttr_win[df_mttr_win['Pole']==pole]['Value'].sum()
                breach = df_breach_win[df_breach_win['Pole']==pole]['Value'].sum()
                taux = breach/total if total else 0
                all_metrics.setdefault(f'SLA_BreachRate_{window}d', {})[pole] = taux

        df_inc = pd.concat([raw_dfs[n] for n in incident_names], ignore_index=True)
        df_chg = pd.concat([raw_dfs[n] for n in change_names], ignore_index=True)
        for window in [7, 30, 90]:
            # Incidents par priorité
            df_inc_win = last_n_days(df_inc, window)
            all_metrics[f'IncidentCounts_{window}d'] = df_inc_win.groupby('Pole')['Value'].sum().to_dict()
            # Changes par typologie
            for typ in ['standard','emergency','failed','cancel','total']:
                typ_names = [n for n in change_names if typ in n.lower()]
                if typ_names:
                    df_typ = pd.concat([raw_dfs[n] for n in typ_names], ignore_index=True)
                    df_typ_win = last_n_days(df_typ, window)
                    all_metrics[f'Change_{typ.capitalize()}_{window}d'] = df_typ_win.groupby('Pole')['Value'].sum().to_dict()
            # Ratio incidents/changes total
            all_metrics[f'IncToChangeRatio_{window}d'] = compute_ratio_incidents_changes(df_inc_win, df_chg).set_index('Pole').to_dict(orient='index')
            # Anomalies incidents/changes
            all_metrics[f'IncidentAnomalies_{window}d'] = detect_anomaly_clusters({'incidents': df_inc_win})
            all_metrics[f'ChangeAnomalies_{window}d'] = detect_anomaly_clusters({'changes': df_chg})
    # Axes résolution incidents : MTTR, SLA
    sla_dfs = [df for df in raw_dfs.values() if 'ResolutionTime' in df.columns]
    if sla_dfs:
        df_sla = sla_dfs[0]
        sla_thresholds = {'P1': 2, 'P2': 4}
        for window in [7, 30, 90]:
            df_sla_win = last_n_days(df_sla, window)
            all_metrics[f'MTTR_{window}d'] = compute_mttr(df_sla_win).set_index(['Pole','Priority']).to_dict(orient='index')
            all_metrics[f'SLACompliance_{window}d'] = compute_sla_compliance(df_sla_win, sla_thresholds).set_index(['Pole','Priority']).to_dict(orient='index')
            all_metrics[f'SLABreachStats_{window}d'] = compute_sla_breach_stats(df_sla_win, sla_thresholds).set_index(['Pole','Priority']).to_dict(orient='index')
            all_metrics[f'MTTRTrends_{window}d'] = compute_mttr_trends(df_sla_win).to_dict(orient='records')
    # Axes résolution incidents : MTTR, SLA
    sla_dfs = [df for df in raw_dfs.values() if 'ResolutionTime' in df.columns]
    if sla_dfs:
        df_sla = sla_dfs[0]
        sla_thresholds = {'P1': 2, 'P2': 4}
        all_metrics['MTTR'] = compute_mttr(df_sla).set_index(['Pole','Priority']).to_dict(orient='index')
        all_metrics['SLACompliance'] = compute_sla_compliance(df_sla, sla_thresholds).set_index(['Pole','Priority']).to_dict(orient='index')
        all_metrics['SLABreachStats'] = compute_sla_breach_stats(df_sla, sla_thresholds).set_index(['Pole','Priority']).to_dict(orient='index')
        all_metrics['MTTRTrends'] = compute_mttr_trends(df_sla).to_dict(orient='records')
    # Axe 11: incidents internes vs externes
    ext_incident_names = [n for n in raw_dfs if 'external' in n.lower()]
    if incident_names and ext_incident_names:
        df_int = pd.concat([raw_dfs[n] for n in incident_names], ignore_index=True)
        df_ext = pd.concat([raw_dfs[n] for n in ext_incident_names], ignore_index=True)
        ext_int_df = compute_external_internal_ratio(df_int, df_ext)
        all_metrics['ExtIntRatio'] = ext_int_df.set_index('Pole').to_dict(orient='index')
    # Axes 12 à 15 : SLA
    sla_dfs = [df for df in raw_dfs.values() if 'ResolutionTime' in df.columns]
    if sla_dfs:
        df_sla = sla_dfs[0]
        sla_thresholds = {'P1': 2, 'P2': 4}
        mttr_df = compute_mttr(df_sla)
        sla_comp_df = compute_sla_compliance(df_sla, sla_thresholds)
        sla_breach_df = compute_sla_breach_stats(df_sla, sla_thresholds)
        mttr_trend_df = compute_mttr_trends(df_sla)
        all_metrics['MTTR'] = mttr_df.set_index(['Pole','Priority']).to_dict(orient='index')
        all_metrics['SLACompliance'] = sla_comp_df.set_index(['Pole','Priority']).to_dict(orient='index')
        all_metrics['SLABreachStats'] = sla_breach_df.set_index(['Pole','Priority']).to_dict(orient='index')
        all_metrics['MTTRTrends'] = mttr_trend_df.to_dict(orient='records')
    # support incidents externes vs internes
    ext_incident_names = [n for n in raw_dfs if 'external' in n.lower()]
    if incident_names and ext_incident_names:
        df_int = pd.concat([raw_dfs[n] for n in incident_names], ignore_index=True)
        df_ext = pd.concat([raw_dfs[n] for n in ext_incident_names], ignore_index=True)
        ext_int_df = compute_external_internal_ratio(df_int, df_ext)
        all_metrics['ExtIntRatio'] = ext_int_df.set_index('Pole').to_dict(orient='index')
    # Analyse intermédiaire LLM par axe d'analyse
    syntheses_axe = {}
    prompts_axe = {}
    # 1. Incidents internes
    if incident_names:
        prompts_axe['incidents'] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents internes (toutes priorités et périodes confondues) :\n{raw_dfs[incident_names[0]].head(10).to_string(index=False)}\n
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
    if change_names:
        prompts_axe['changes'] = f"""
Vous êtes un expert IT analyste. Voici les données de changes (toutes typologies et périodes confondues) :\n{raw_dfs[change_names[0]].head(10).to_string(index=False)}\n
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
    if mttr_names:
        prompts_axe['mttr'] = f"""
Vous êtes un expert IT analyste. Voici les données de temps moyen de résolution (MTTR) :\n{raw_dfs[mttr_names[0]].head(10).to_string(index=False)}\n
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
    if breach_sla_names:
        prompts_axe['sla'] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents hors SLA :\n{raw_dfs[breach_sla_names[0]].head(10).to_string(index=False)}\n
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
    if external_names:
        prompts_axe['externes'] = f"""
Vous êtes un expert IT analyste. Voici les données d'incidents externes :\n{raw_dfs[external_names[0]].head(10).to_string(index=False)}\n
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
    prompt_global = generate_global_prompt(all_metrics, syntheses_axe)
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
    prompt_cto = generate_global_prompt(all_metrics)
    print("\n=== PROMPT CTO POUR SYSTEME LLM/IA ===\n")
    print(prompt_cto)

if __name__ == '__main__':
    main()
