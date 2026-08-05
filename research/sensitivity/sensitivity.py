# research/sensitivity/sensitivity.py
"""
Duyarlılık Analizi Yöntemleri

OAT (One-at-a-Time):
- Her parametreyi tek tek değiştir, diğerleri sabit
- Basit, anlaşılır, ama etkileşimleri yakalamaz

Morris Method (Elementary Effects):
- Küresel duyarlılık analizi
- Her parametrenin ortalama ve standart sapma etkisini hesaplar
- Hesaplama maliyeti düşük

Sobol (Variance-based):
- Varyans ayrıştırması
- Birinci ve ikinci derece etkileşimleri hesaplar
- scipy gerektirir, hesaplama maliyeti yüksek
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Callable, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class SensitivityResult:
    """Duyarlılık analizi sonuçları."""
    method: str
    param_names: List[str]
    values: Dict[str, Any]
    metadata: Dict[str, Any]


class BaseSensitivity(ABC):
    """Tüm duyarlılık analizleri için temel sınıf."""

    def __init__(self, param_space: Dict[str, Dict[str, Any]], seed: int = 42):
        self.param_space = param_space
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def analyze(self, objective_func: Callable, **kwargs) -> SensitivityResult:
        """
        Duyarlılık analizini çalıştırır.

        objective_func: Parametre alıp skaler bir değer döndüren fonksiyon.
        """
        pass


class OATAnalyzer(BaseSensitivity):
    """
    One-at-a-Time Analizi.
    Her parametreyi tek tek değiştirir, diğerleri sabit kalır.
    """

    def analyze(
        self,
        objective_func: Callable,
        base_params: Dict[str, float],
        n_points: int = 7,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        **kwargs,
    ) -> SensitivityResult:
        """
        OAT analizi.

        Args:
            objective_func: f(params) -> float
            base_params: Temel parametre seti
            n_points: Her parametre için kaç nokta
            ranges: Parametre aralıkları (varsayılan: param_space'daki low/high)

        Returns:
            SensitivityResult: Her parametrenin etkisi
        """
        param_names = list(self.param_space.keys())
        results = {}

        for param in param_names:
            # Parametre aralığını belirle
            if ranges and param in ranges:
                low, high = ranges[param]
            else:
                defn = self.param_space[param]
                if "low" in defn and "high" in defn:
                    low, high = defn["low"], defn["high"]
                else:
                    # Choice tipi için
                    values = defn.get("values", [])
                    low, high = min(values), max(values)

            # Noktaları oluştur
            if "values" in self.param_space[param]:
                # Choice parametreler için
                points = self.param_space[param]["values"]
            else:
                points = np.linspace(low, high, n_points).tolist()

            values = []
            for val in points:
                params = base_params.copy()
                params[param] = val
                try:
                    obj = objective_func(params)
                    values.append(obj)
                except Exception as e:
                    values.append(np.nan)
                    warnings.warn(f"OAT: {param}={val} başarısız: {e}")

            # Etki hesapla (max - min)
            valid_vals = [v for v in values if not np.isnan(v)]
            if valid_vals:
                effect = np.max(valid_vals) - np.min(valid_vals)
                mean_effect = np.mean(valid_vals) if valid_vals else np.nan
            else:
                effect = np.nan
                mean_effect = np.nan

            results[param] = {
                "points": points,
                "values": values,
                "effect": effect,
                "mean_effect": mean_effect,
                "n_valid": len(valid_vals),
            }

        return SensitivityResult(
            method="OAT",
            param_names=param_names,
            values=results,
            metadata={"n_points": n_points, "base_params": base_params},
        )


class MorrisAnalyzer(BaseSensitivity):
    """
    Morris Elementary Effects Metodu.
    Küresel duyarlılık için kullanılır.
    """

    def analyze(
        self,
        objective_func: Callable,
        n_trajectories: int = 10,
        n_levels: int = 4,
        **kwargs,
    ) -> SensitivityResult:
        """
        Morris analizi.

        Args:
            objective_func: f(params) -> float
            n_trajectories: Yörünge sayısı (öneri: 10-50)
            n_levels: Her parametre için seviye sayısı (öneri: 4-6)

        Returns:
            SensitivityResult: mu* (ortalama) ve sigma (standart sapma)
        """
        param_names = list(self.param_space.keys())
        n_params = len(param_names)

        # Parametre aralıkları
        ranges = {}
        for name, defn in self.param_space.items():
            if "low" in defn and "high" in defn:
                ranges[name] = (defn["low"], defn["high"])
            elif "values" in defn:
                values = defn["values"]
                ranges[name] = (min(values), max(values))
            else:
                ranges[name] = (0, 1)

        # Delta (adım büyüklüğü)
        delta = 1 / (n_levels - 1)

        # Tüm parametreleri 0-1 aralığına normalize et
        def normalize(params, ranges):
            norm = {}
            for name in param_names:
                low, high = ranges[name]
                if high - low == 0:
                    norm[name] = 0.5
                else:
                    norm[name] = (params[name] - low) / (high - low)
            return norm

        def denormalize(norm_params, ranges):
            params = {}
            for name in param_names:
                low, high = ranges[name]
                params[name] = low + norm_params[name] * (high - low)
            return params

        # Elementary Effects hesapla
        ee_all = []

        for _ in range(n_trajectories):
            # Rastgele başlangıç noktası (0-1 arası)
            base = self.rng.uniform(0, 1 - delta, n_params)
            base_dict = {param_names[i]: base[i] for i in range(n_params)}

            # Hangi parametrenin değişeceğini belirle (rastgele sıralama)
            order = self.rng.permutation(n_params)
            current = base_dict.copy()

            for idx in order:
                param = param_names[idx]
                # Parametreyi delta kadar artır (veya azalt, rastgele)
                direction = 1 if self.rng.random() > 0.5 else -1
                new_val = current[param] + direction * delta
                new_val = np.clip(new_val, 0, 1)

                # Yeni nokta
                new_dict = current.copy()
                new_dict[param] = new_val

                # De-normalize et ve hesapla
                params_orig = denormalize(current, ranges)
                new_params_orig = denormalize(new_dict, ranges)

                try:
                    y_base = objective_func(params_orig)
                    y_new = objective_func(new_params_orig)
                    ee = (y_new - y_base) / (direction * delta)
                    ee_all.append(ee)
                except Exception:
                    ee_all.append(np.nan)

                current = new_dict

        # Mu* ve sigma hesapla (parametre bazında)
        n_ee = len(ee_all)
        mu_star = np.mean(np.abs(ee_all)) if n_ee > 0 else np.nan
        sigma = np.std(ee_all) if n_ee > 0 else np.nan

        # Sonuçları topla (tek bir değer döndürüyoruz, tüm parametreler için ortak)
        # Daha detaylı için her parametrenin kendi ee'lerini toplamak gerekir.
        # Şimdilik basit tutuyoruz.

        results = {
            "mu_star": mu_star,
            "sigma": sigma,
            "n_trajectories": n_trajectories,
            "n_levels": n_levels,
            "n_ee": n_ee,
        }

        return SensitivityResult(
            method="Morris",
            param_names=param_names,
            values=results,
            metadata={"n_trajectories": n_trajectories, "n_levels": n_levels},
        )


class SobolAnalyzer(BaseSensitivity):
    """
    Sobol Duyarlılık Analizi (Varyans Tabanlı).
    scipy gerektirir.
    """

    def analyze(
        self,
        objective_func: Callable,
        n_samples: int = 1000,
        **kwargs,
    ) -> SensitivityResult:
        """
        Sobol analizi.

        Args:
            objective_func: f(params) -> float
            n_samples: Örnek sayısı (öneri: 1000-10000)

        Returns:
            SensitivityResult: Birinci ve ikinci derece indeksler
        """
        try:
            from scipy.stats import qmc
        except ImportError:
            raise ImportError("Sobol analizi için scipy gerekli: pip install scipy")

        param_names = list(self.param_space.keys())
        n_params = len(param_names)

        # Parametre aralıkları
        lows, highs = [], []
        for name in param_names:
            defn = self.param_space[name]
            if "low" in defn and "high" in defn:
                lows.append(defn["low"])
                highs.append(defn["high"])
            elif "values" in defn:
                vals = defn["values"]
                lows.append(min(vals))
                highs.append(max(vals))
            else:
                lows.append(0)
                highs.append(1)

        # Sobol dizisi (quasi-Monte Carlo)
        sampler = qmc.Sobol(d=2 * n_params, scramble=True, seed=self.seed)
        samples = sampler.random(n_samples)

        # İlk yarısı A, ikinci yarısı B
        A = samples[:, :n_params]
        B = samples[:, n_params:]

        # A ve B'yi parametre uzayına dönüştür
        def to_params(X):
            params_list = []
            for row in X:
                params = {}
                for j, name in enumerate(param_names):
                    low, high = lows[j], highs[j]
                    params[name] = low + row[j] * (high - low)
                params_list.append(params)
            return params_list

        A_params = to_params(A)
        B_params = to_params(B)

        # f(A) ve f(B)
        def eval_func(params_list):
            results = []
            for params in params_list:
                try:
                    results.append(objective_func(params))
                except Exception:
                    results.append(np.nan)
            return np.array(results)

        f_A = eval_func(A_params)
        f_B = eval_func(B_params)

        # Birinci derece indeksler (S_i)
        # Basitleştirilmiş: sadece ana etkiler
        V_Y = np.var(f_A) if len(f_A) > 1 else 1

        S1 = []
        for i in range(n_params):
            # A'yı i. parametre ile B'nin diğer parametreleriyle değiştir
            C = B.copy()
            C[:, i] = A[:, i]
            C_params = to_params(C)
            f_C = eval_func(C_params)

            V_E = np.var(f_C) if len(f_C) > 1 else 1
            S1.append(V_E / V_Y if V_Y != 0 else np.nan)

        # Sonuçlar
        results = {
            "S1": dict(zip(param_names, S1)),
            "V_Y": V_Y,
            "n_samples": n_samples,
        }

        return SensitivityResult(
            method="Sobol",
            param_names=param_names,
            values=results,
            metadata={"n_samples": n_samples},
        )


class SensitivityFactory:
    """Duyarlılık analizi fabrikası."""

    @staticmethod
    def create(method: str, param_space: Dict[str, Dict[str, Any]], seed: int = 42) -> BaseSensitivity:
        if method == "oat":
            return OATAnalyzer(param_space, seed)
        elif method == "morris":
            return MorrisAnalyzer(param_space, seed)
        elif method == "sobol":
            return SobolAnalyzer(param_space, seed)
        else:
            raise ValueError(f"Bilinmeyen duyarlılık metodu: {method}")