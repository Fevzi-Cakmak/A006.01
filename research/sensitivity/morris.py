"""
Morris Metodu (Elementary Effects / Screening)
Hangi parametrelerin etkili olduğunu az sayıda örnekle belirler.
Sobol'dan daha hızlıdır, ilk eleme için kullanılır.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logger.warning("SALib yüklü değil. Morris analizi kullanılamaz. (pip install SALib)")


def compute_morris_indices(
    df: pd.DataFrame,
    target_col: str,
    param_cols: List[str],
    num_levels: int = 4,
    n_trajectories: int = 20,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Morris Elementary Effects indekslerini hesaplar.
    
    Args:
        df: Veri seti (en az 10 satır olmalı)
        target_col: Hedef değişken (örn. compound_return)
        param_cols: Parametre sütunları
        num_levels: Her parametre için seviye sayısı (4 veya 6 önerilir)
        n_trajectories: Trajectory sayısı (arttıkça doğruluk artar, süre uzar)
        random_state: Rastgelelik tohumu
    
    Returns:
        pd.DataFrame: Parametre başına mu, mu_star, sigma içerir
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib yüklü değil. pip install SALib")
    
    if len(df) < 10:
        logger.warning("Morris analizi için yeterli veri yok (en az 10 gözlem gerekli).")
        return pd.DataFrame()
    
    # Parametre aralıklarını tanımla
    problem = {
        'num_vars': len(param_cols),
        'names': param_cols,
        'bounds': [
            [df[col].min(), df[col].max()] for col in param_cols
        ]
    }
    
    # Morris örneklemesi yap (SALib) - grid_jump kaldırıldı
    np.random.seed(random_state)
    param_values = morris_sample.sample(
        problem, 
        N=n_trajectories,
        num_levels=num_levels
    )
    
    # Örneklenen parametreler üzerinden tahmin yapmak için Random Forest kullan
    from sklearn.ensemble import RandomForestRegressor
    
    X_train = df[param_cols].values
    y_train = df[target_col].values
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Morris örneklemleri üzerinden tahmin yap
    y_pred = model.predict(param_values)
    
    # Morris indekslerini hesapla (güncel SALib için)
    Si = morris_analyze.analyze(
        problem, 
        param_values, 
        y_pred,
        num_levels=num_levels
    )
    
    # Sonuçları DataFrame'e dönüştür
    results = pd.DataFrame({
        'parametre': param_cols,
        'mu': Si['mu'],
        'mu_star': Si['mu_star'],  # En güvenilir metrik (mutlak ortalama)
        'sigma': Si['sigma'],
    })
    
    # mu_star'a göre sırala (en etkiliden en az etkiliye)
    results = results.sort_values('mu_star', ascending=False)
    
    logger.info(f"Morris analizi tamamlandı. En etkili parametre: {results.iloc[0]['parametre']} (mu_star={results.iloc[0]['mu_star']:.3f})")
    
    return results