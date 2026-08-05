import random
import numpy as np
from typing import Dict, Any, List
from itertools import product
from .base import BaseSampler


class GridSampler(BaseSampler):
    """Grid Search (Parametre uzayını tara)."""

    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        ranges = self._get_param_ranges()
        param_keys = list(ranges.keys())
        n_params = len(param_keys)

        # Her boyut için kaç nokta kullanılacağını hesapla
        # En az 2, en fazla num_samples^(1/n_params) kadar
        points_per_dim = max(2, int(round(num_samples ** (1.0 / n_params))))

        grid_combinations = []
        for key in param_keys:
            val = ranges[key]
            if isinstance(val, tuple):
                # Sürekli: eşit aralıklı noktalar
                grid = np.linspace(val[0], val[1], points_per_dim).tolist()
            elif isinstance(val, list):
                # Kategorik: doğrudan liste
                grid = val
            else:
                # Sabit
                grid = [val]
            grid_combinations.append(grid)

        # Tüm kombinasyonları oluştur
        all_combos = list(product(*grid_combinations))

        # Eğer çok fazla kombinasyon varsa rastgele örnek al
        if len(all_combos) > num_samples:
            random.seed(self.random_state)
            all_combos = random.sample(all_combos, num_samples)

        # Sözlük listesine dönüştür
        result = []
        for combo in all_combos:
            sample = {param_keys[i]: combo[i] for i in range(len(param_keys))}
            result.append(sample)
        return result