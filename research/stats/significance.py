import pandas as pd
import numpy as np
from scipy import stats

def correlation_with_ci(
    df: pd.DataFrame,
    col1: str,
    col2: str,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict:
    x = df[col1].dropna()
    y = df[col2].dropna()
    valid_idx = x.index.intersection(y.index)
    x = x.loc[valid_idx]
    y = y.loc[valid_idx]
    if len(x) < 3:
        return {'error': 'Yetersiz gözlem'}
    
    # 1. Pearson
    pearson_r, pearson_p = stats.pearsonr(x, y)
    # 2. Spearman
    spearman_r, spearman_p = stats.spearmanr(x, y)
    # 3. Kendall Tau
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    # 4. Bootstrap CI - default_rng ile
    rng = np.random.default_rng(random_state)
    n = len(x)
    boot_r = np.zeros(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_r[i] = stats.pearsonr(x.iloc[idx], y.iloc[idx])[0]
    
    alpha = (1 - confidence_level) / 2
    ci_lower = np.percentile(boot_r, alpha * 100)
    ci_upper = np.percentile(boot_r, (1 - alpha) * 100)
    
    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'kendall_tau': kendall_tau,
        'kendall_p': kendall_p,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
    }

def batch_correlation_analysis(
    df: pd.DataFrame,
    pairs: list,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> pd.DataFrame:
    results = []
    for c1, c2 in pairs:
        if c1 not in df.columns or c2 not in df.columns:
            continue
        res = correlation_with_ci(df, c1, c2, n_bootstrap, random_state=random_state)
        if 'error' in res:
            continue
        results.append({
            'pair': f'{c1} vs {c2}',
            'pearson_r': res['pearson_r'],
            'pearson_p': res['pearson_p'],
            'spearman_r': res['spearman_r'],
            'spearman_p': res['spearman_p'],
            'kendall_tau': res['kendall_tau'],
            'kendall_p': res['kendall_p'],
            'ci_%95_lower': res['ci_lower'],
            'ci_%95_upper': res['ci_upper'],
        })
    return pd.DataFrame(results)