import numpy as np
from typing import Dict, Any, List
from .base import BaseSampler


class RandomSampler(BaseSampler):
    """Rastgele uniform örnekleme (Monte Carlo)."""

    def generate(self, num_samples: int) -> List[Dict[str, Any]]:
        np.random.seed(self.random_state)
        samples = []

        ranges = self._get_param_ranges()
        param_keys = list(ranges.keys())

        for _ in range(num_samples):
            sample = {}
            for key in param_keys:
                val = ranges[key]
                if isinstance(val, tuple):
                    # Sürekli değişken
                    sample[key] = np.random.uniform(val[0], val[1])
                elif isinstance(val, list) and all(isinstance(i, (int, str)) for i in val):
                    # Kategorik değişken
                    sample[key] = np.random.choice(val)
                else:
                    # Sabit
                    sample[key] = val
            samples.append(sample)
        
        return samples