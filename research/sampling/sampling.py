# research/sampling/sampling.py
"""
Örnekleme stratejileri:

- Random Sampler: Tamamen rastgele
- Latin Hypercube: Daha homojen dağılım
- Sobol Sampler: Düşük sapmalı dizi (quasi-Monte Carlo)
- Grid Sampler: Izgara (taramalı) örnekleme
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple


class BaseSampler(ABC):
    """Tüm örnekleyiciler için temel sınıf."""

    def __init__(self, param_space: Dict[str, Dict[str, Any]], seed: int = 42):
        """
        param_space: {
            'stop_loss_ratio': {'low': 0.08, 'high': 0.12, 'dist': 'uniform'},
            'adx_strong': {'values': [28, 30, 32, 35], 'dist': 'choice'},
            'rsi_low': {'low': 55, 'high': 65, 'dist': 'uniform'},
        }
        """
        self.param_space = param_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def sample(self, n: int) -> List[Dict[str, Any]]:
        """n adet parametre seti üretir."""
        pass

    def _sample_single_param(self, param_name: str, param_def: Dict[str, Any]) -> np.ndarray:
        """Tek bir parametre için örnekleme yapar (vektörel)."""
        dist = param_def.get("dist", "uniform")
        n = param_def.get("n", 1)  # Bu metod artık n'yi dışarıdan alacak, burada kullanma

    def _sample_single_value(self, param_def: Dict[str, Any]) -> float:
        """Tek bir değer örnekler."""
        dist = param_def.get("dist", "uniform")
        if dist == "uniform":
            low = param_def["low"]
            high = param_def["high"]
            return self.rng.uniform(low, high)
        elif dist == "normal":
            loc = param_def.get("loc", 0)
            scale = param_def.get("scale", 1)
            low = param_def.get("low", -np.inf)
            high = param_def.get("high", np.inf)
            val = self.rng.normal(loc, scale)
            return np.clip(val, low, high)
        elif dist == "choice":
            values = param_def["values"]
            return self.rng.choice(values)
        elif dist == "lognormal":
            mean = param_def.get("mean", 0)
            sigma = param_def.get("sigma", 1)
            low = param_def.get("low", 0)
            high = param_def.get("high", np.inf)
            val = self.rng.lognormal(mean, sigma)
            return np.clip(val, low, high)
        else:
            raise ValueError(f"Bilinmeyen dağılım: {dist}")


class RandomSampler(BaseSampler):
    """Tamamen rastgele örnekleme."""

    def sample(self, n: int) -> List[Dict[str, Any]]:
        results = []
        for _ in range(n):
            params = {}
            for name, defn in self.param_space.items():
                params[name] = self._sample_single_value(defn)
            results.append(params)
        return results


class LatinHypercubeSampler(BaseSampler):
    """
    Latin Hypercube Sampling (LHS) - daha homojen dağılım.
    Sürekli parametreler için uygundur.
    """

    def sample(self, n: int) -> List[Dict[str, Any]]:
        param_names = list(self.param_space.keys())
        n_params = len(param_names)
        
        # LHS matrisi (0-1 arası)
        lhs_matrix = np.zeros((n, n_params))
        for i in range(n_params):
            perm = self.rng.permutation(n)
            lhs_matrix[:, i] = (perm + self.rng.uniform(0, 1, n)) / n
        
        # Parametre değerlerine dönüştür
        results = []
        for i in range(n):
            params = {}
            for j, name in enumerate(param_names):
                defn = self.param_space[name]
                u = lhs_matrix[i, j]
                # Uniform quantile dönüşümü
                dist = defn.get("dist", "uniform")
                if dist == "uniform":
                    low = defn["low"]
                    high = defn["high"]
                    val = low + u * (high - low)
                elif dist == "normal":
                    loc = defn.get("loc", 0)
                    scale = defn.get("scale", 1)
                    low = defn.get("low", -np.inf)
                    high = defn.get("high", np.inf)
                    val = np.clip(self.rng.normal(loc, scale), low, high)
                    # LHS için normal dağılımı tam desteklemiyoruz, basitçe rastgele al
                    # Daha iyisi: ppf kullanmak ama şimdilik basit
                    val = self.rng.normal(loc, scale)
                    val = np.clip(val, low, high)
                elif dist == "choice":
                    values = defn["values"]
                    idx = int(u * len(values)) % len(values)
                    val = values[idx]
                else:
                    val = self._sample_single_value(defn)
                params[name] = val
            results.append(params)
        return results


class SobolSampler(BaseSampler):
    """
    Sobol dizisi (quasi-Monte Carlo) - düşük sapmalı.
    scipy gerektirir.
    """

    def sample(self, n: int) -> List[Dict[str, Any]]:
        try:
            from scipy.stats import qmc
        except ImportError:
            raise ImportError("SobolSampler için scipy gerekli: pip install scipy")
        
        param_names = list(self.param_space.keys())
        n_params = len(param_names)
        
        # Sobol dizisi
        sampler = qmc.Sobol(d=n_params, scramble=True, seed=self.seed)
        sobol_samples = sampler.random(n)
        
        results = []
        for i in range(n):
            params = {}
            for j, name in enumerate(param_names):
                defn = self.param_space[name]
                u = sobol_samples[i, j]
                dist = defn.get("dist", "uniform")
                if dist == "uniform":
                    low = defn["low"]
                    high = defn["high"]
                    val = low + u * (high - low)
                elif dist == "choice":
                    values = defn["values"]
                    idx = int(u * len(values)) % len(values)
                    val = values[idx]
                else:
                    val = self._sample_single_value(defn)
                params[name] = val
            results.append(params)
        return results


class GridSampler(BaseSampler):
    """
    Izgara (grid) taraması. Tüm kombinasyonları dener.
    Parametre sayısı arttıkça kombinasyon sayısı patlar, dikkatli kullanın.
    """

    def sample(self, n: int = None) -> List[Dict[str, Any]]:
        # n parametresi grid için anlamsız, tüm kombinasyonları döndürür
        import itertools
        
        param_names = list(self.param_space.keys())
        all_values = []
        for name in param_names:
            defn = self.param_space[name]
            if defn.get("dist") == "choice":
                values = defn["values"]
            else:
                low = defn.get("low", 0)
                high = defn.get("high", 1)
                # Varsayılan 5 nokta
                values = np.linspace(low, high, 5).tolist()
            all_values.append(values)
        
        combinations = list(itertools.product(*all_values))
        results = []
        for combo in combinations:
            params = {}
            for i, name in enumerate(param_names):
                params[name] = combo[i]
            results.append(params)
        return results


class SamplerFactory:
    """Örnekleyici fabrikası."""

    @staticmethod
    def create(sampler_type: str, param_space: Dict[str, Dict[str, Any]], seed: int = 42) -> BaseSampler:
        if sampler_type == "random":
            return RandomSampler(param_space, seed)
        elif sampler_type == "latin_hypercube":
            return LatinHypercubeSampler(param_space, seed)
        elif sampler_type == "sobol":
            return SobolSampler(param_space, seed)
        elif sampler_type == "grid":
            return GridSampler(param_space, seed)
        else:
            raise ValueError(f"Bilinmeyen sampler tipi: {sampler_type}")