"""
Sobol Duyarlılık Analizi (SALib ile)
Hangi parametrelerin getiri üzerindeki varyansa en çok katkı yaptığını hesaplar.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    from SALib.sample import sobol as sobol_sample
    from SALib.analyze import sobol as sobol_analyze
    SALIB_AVAILABLE = True
except ImportError:
    SALIB_AVAILABLE = False
    logger.warning("SALib yüklü değil. Sobol analizi kullanılamaz. (pip install SALib)")


def compute_sobol_indices(
    df: pd.DataFrame,
    target_col: str,
    param_cols: List[str],
    n_samples: int = 1024,
    random_state: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Sobol duyarlılık indekslerini hesaplar.
    
    Args:
        df: Veri seti
        target_col: Hedef değişken (örn. compound_return)
        param_cols: Parametre sütunları
        n_samples: Sobol örnek sayısı (SALib'de N*2D şeklinde çalışır)
        random_state: Rastgelelik tohumu
    
    Returns:
        pd.DataFrame: Parametre başına S1, S1_conf, ST, ST_conf içerir
    """
    if not SALIB_AVAILABLE:
        raise ImportError("SALib yüklü değil. pip install SALib")
    
    if len(df) < 10:
        logger.warning("Sobol analizi için yeterli veri yok (en az 10 gözlem gerekli).")
        return pd.DataFrame()
    
    # Parametre aralıklarını hesapla
    problem = {
        'num_vars': len(param_cols),
        'names': param_cols,
        'bounds': [
            [df[col].min(), df[col].max()] for col in param_cols
        ]
    }
    
    # Sobol örneklemesi yap (SALib)
    np.random.seed(random_state)
    param_values = sobol_sample.sample(problem, n_samples, calc_second_order=False)
    
    # Örneklenen parametreler üzerinden tahmin yapmak için basit bir model kur
    # Burada Random Forest kullanıyoruz (importance modülü ile aynı)
    from sklearn.ensemble import RandomForestRegressor
    
    X_train = df[param_cols].values
    y_train = df[target_col].values
    
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Sobol örneklemleri üzerinden tahmin yap
    y_pred = model.predict(param_values)
    
    # Sobol indekslerini hesapla
    Si = sobol_analyze.analyze(problem, y_pred, calc_second_order=False, print_to_console=False)
    
    # Sonuçları DataFrame'e dönüştür
    results = pd.DataFrame({
        'parametre': param_cols,
        'S1': Si['S1'],
        'S1_conf': Si['S1_conf'],
        'ST': Si['ST'],
        'ST_conf': Si['ST_conf'],
    })
    
    # ST'ye göre sırala (toplam etki)
    results = results.sort_values('ST', ascending=False)
    
    logger.info(f"Sobol analizi tamamlandı. En etkili parametre: {results.iloc[0]['parametre']} (ST={results.iloc[0]['ST']:.3f})")
    
    return results