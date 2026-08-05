from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd


class BaseSampler(ABC):
    """Parametre örnekleme stratejileri için temel sınıf."""

    def __init__(self, param_space: Dict[str, Any], random_state: int = 42):
        """
        Args:
            param_space: Örnekleme uzayı. 
                         Örn: {'rsi_min': (50, 70), 'stop_loss': (0.05, 0.15)}
            random_state: Rastgelelik için seed.
        """
        self.param_space = param_space
        self.random_state = random_state

    @abstractmethod
    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        """
        Belirtilen sayıda parametre kombinasyonu üretir.

        Returns:
            Parametre sözlüklerinin listesi.
        """
        pass

    def _get_param_ranges(self) -> Dict[str, tuple]:
        """Parametre aralıklarını döndürür (validation için)."""
        ranges = {}
        for key, value in self.param_space.items():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                ranges[key] = (float(value[0]), float(value[1]))
            else:
                # Kategorik veya sabit değerler için
                ranges[key] = value
        return ranges