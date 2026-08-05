from abc import ABC, abstractmethod
import pandas as pd


class BaseImportanceAnalyzer(ABC):
    """Parametre önem derecelendirme stratejileri için temel sınıf."""

    @abstractmethod
    def analyze(self, results_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """
        Sonuç dataframe'ini analiz eder ve parametre önem skorlarını döndürür.

        Args:
            results_df: Her satırı bir simülasyon olan DataFrame.
                       Sütunlar: Parametreler + 'target_col'
            target_col: Hedef metrik sütun adı.

        Returns:
            DataFrame: ['parametre', 'importance_score', 'rank'] sütunlarıyla.
        """
        pass