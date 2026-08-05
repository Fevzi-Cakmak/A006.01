import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

def compute_permutation_importance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    n_repeats: int = 10,
    n_estimators: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    X = df[feature_cols]
    y = df[target_col]
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X, y)
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'permutation_mean': result.importances_mean,
        'permutation_std': result.importances_std,
    }).sort_values('permutation_mean', ascending=False)
    return imp_df