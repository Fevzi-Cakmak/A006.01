import numpy as np
from typing import Dict, Any, List
from .base import BaseSampler


class LatinHypercubeSampler(BaseSampler):
    """Latin Hypercube Örnekleme (Daha dengeli dağılım)."""

    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        np.random.seed(self.random_state)
        ranges = self._get_param_ranges()
        param_keys = list(ranges.keys())
        n_params = len(param_keys)

        # [0,1) aralığında Latin Hypercube matrisi oluştur
        samples = np.random.uniform(0, 1, size=(num_samples, n_params))
        for i in range(n_params):
            perm = np.random.permutation(num_samples)
            samples[:, i] = (perm + samples[:, i]) / num_samples

        # Gerçek parametre aralıklarına dönüştür
        result = []
        for i in range(num_samples):
            sample = {}
            for j, key in enumerate(param_keys):
                val = ranges[key]
                if isinstance(val, tuple):
                    # Sürekli değişken
                    sample[key] = val[0] + samples[i, j] * (val[1] - val[0])
                elif isinstance(val, list) and all(isinstance(v, (int, float, str)) for v in val):
                    # Kategorik - değeri indexe göre seç
                    idx = int(np.floor(samples[i, j] * len(val)))
                    sample[key] = val[min(idx, len(val) - 1)]
                else:
                    # Sabit değer
                    sample[key] = val
            result.append(sample)
        return result