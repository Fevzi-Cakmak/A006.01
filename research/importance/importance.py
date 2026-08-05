# research/importance/importance.py
"""
Parametre önem analizi metodları:

- Pearson: Doğrusal korelasyon
- Spearman: Sıralama bazlı (monotonik) korelasyon
- Kendall: Sıralama bazlı (daha sağlam)
- Random Forest: Ağaç tabanlı önem
- Permutation Importance: Model-agnostik önem
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


class ImportanceAnalyzer(ABC):
    """Tüm önem analizcileri için temel sınıf."""

    def __init__(self, name: str = "Base"):
        self.name = name

    @abstractmethod
    def analyze(self, df: pd.DataFrame, target_col: str = "getiri") -> pd.Series:
        """Önem skorlarını hesaplar."""
        pass

    def get_name(self) -> str:
        return self.name


class PearsonImportance(ImportanceAnalyzer):
    """Pearson korelasyonu (doğrusal ilişki)."""

    def __init__(self):
        super().__init__("Pearson")

    def analyze(self, df: pd.DataFrame, target_col: str = "getiri") -> pd.Series:
        features = [c for c in df.columns if c != target_col]
        corr = df[features + [target_col]].corr(method="pearson")[target_col].drop(target_col)
        return corr.abs().sort_values(ascending=False)


class SpearmanImportance(ImportanceAnalyzer):
    """Spearman sıralama korelasyonu (monotonik ilişki)."""

    def __init__(self):
        super().__init__("Spearman")

    def analyze(self, df: pd.DataFrame, target_col: str = "getiri") -> pd.Series:
        features = [c for c in df.columns if c != target_col]
        corr = df[features + [target_col]].corr(method="spearman")[target_col].drop(target_col)
        return corr.abs().sort_values(ascending=False)


class KendallImportance(ImportanceAnalyzer):
    """Kendall sıralama korelasyonu (daha sağlam)."""

    def __init__(self):
        super().__init__("Kendall")

    def analyze(self, df: pd.DataFrame, target_col: str = "getiri") -> pd.Series:
        features = [c for c in df.columns if c != target_col]
        corr = df[features + [target_col]].corr(method="kendall")[target_col].drop(target_col)
        return corr.abs().sort_values(ascending=False)


class RandomForestImportance(ImportanceAnalyzer):
    """Random Forest ile önem skorları."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        super().__init__(f"RandomForest(n={n_estimators})")
        self.n_estimators = n_estimators
        self.random_state = random_state

    def analyze(self, df: pd.DataFrame, target_col: str = "getiri") -> pd.Series:
        features = [c for c in df.columns if c != target_col]
        X = df[features].fillna(0)
        y = df[target_col].fillna(0)
        
        # NaN veya inf kontrolü
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importance = pd.Series(rf.feature_importances_, index=features)
        return importance.sort_values(ascending=False)


class PermutationImportance(ImportanceAnalyzer):
    """Permutation importance (model-agnostik)."""

    def __init__(self, n_repeats: int = 10, random_state: int = 42, n_estimators: int = 100):
        super().__init__(f"Permutation(n={n_repeats})")
        self.n_repeats = n_repeats
        self.random_state = random_state
        self.n_estimators = n_estimators

    def analyze(self, df: pd.DataFrame, target_col: str = "getiri") -> pd.Series:
        features = [c for c in df.columns if c != target_col]
        X = df[features].fillna(0)
        y = df[target_col].fillna(0)
        
        # NaN veya inf kontrolü
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        result = permutation_importance(
            rf, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        importance = pd.Series(result.importances_mean, index=features)
        return importance.sort_values(ascending=False)


class ImportanceFactory:
    """Önem analizcisi fabrikası."""

    @staticmethod
    def create(method: str, **kwargs) -> ImportanceAnalyzer:
        if method == "pearson":
            return PearsonImportance()
        elif method == "spearman":
            return SpearmanImportance()
        elif method == "kendall":
            return KendallImportance()
        elif method == "random_forest":
            return RandomForestImportance(
                n_estimators=kwargs.get("n_estimators", 100),
                random_state=kwargs.get("random_state", 42)
            )
        elif method == "permutation":
            return PermutationImportance(
                n_repeats=kwargs.get("n_repeats", 10),
                random_state=kwargs.get("random_state", 42),
                n_estimators=kwargs.get("n_estimators", 100)
            )
        else:
            raise ValueError(f"Bilinmeyen önem metodu: {method}")