import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def compute_rf_importance(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
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
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'rf_mdi_importance': model.feature_importances_
    }).sort_values('rf_mdi_importance', ascending=False)
    return imp_df