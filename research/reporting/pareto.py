import pandas as pd

def compute_multi_criteria_pareto(
    df: pd.DataFrame,
    top_n: int = 20,
    weights: tuple = (0.4, 0.3, 0.3),
) -> pd.DataFrame:
    df_copy = df.copy()
    df_copy['loss_score'] = -df_copy['max_loss']
    
    # ASCII uyumlu sütun adı kullan
    norm_getiri = df_copy['compound_return'] / df_copy['compound_return'].max()
    norm_pf = df_copy['profit_factor'] / df_copy['profit_factor'].max()
    norm_loss = df_copy['loss_score'] / df_copy['loss_score'].max()
    
    w1, w2, w3 = weights
    df_copy['composite_score'] = (w1 * norm_getiri) + (w2 * norm_pf) + (w3 * norm_loss)
    
    top = df_copy.nlargest(top_n, 'composite_score')
    summary = top[['compound_return', 'avg_return', 'profit_factor', 'win_rate', 'max_loss']].mean()
    summary_all = df[['compound_return', 'avg_return', 'profit_factor', 'win_rate', 'max_loss']].mean()
    
    result_df = pd.DataFrame({
        'Metrik': ['compound_return', 'avg_return', 'profit_factor', 'win_rate', 'max_loss'],
        'Top20_Ortalama': summary.values,
        'TumVeri_Ortalama': summary_all.values,
        'Fark': summary.values - summary_all.values,
    })
    return result_df