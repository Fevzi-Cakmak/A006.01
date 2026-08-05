# research/stats/stats.py
"""
İstatistiksel metrik hesaplamaları.

Tüm fonksiyonlar NumPy dizileri veya listeler üzerinde çalışır.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any
from scipy import stats as scipy_stats


@dataclass
class StatResult:
    """İstatistik sonuçları için veri sınıfı."""
    metric: str
    value: float
    metadata: Dict[str, Any]


def calculate_bias(simulated: Union[np.ndarray, List[float]], original: float) -> float:
    """
    Bootstrap bias hesaplar.
    
    Bias = mean(simulated) - original
    
    Args:
        simulated: Bootstrap simülasyon sonuçları
        original: Orijinal değer
    
    Returns:
        float: Bias değeri
    """
    sim = np.array(simulated)
    return np.mean(sim) - original


def calculate_var(
    returns: Union[np.ndarray, List[float]], 
    confidence: float = 0.95,
    method: str = "historical"
) -> float:
    """
    Value at Risk (VaR) hesaplar.
    
    Args:
        returns: Getiri dizisi
        confidence: Güven seviyesi (varsayılan: 0.95)
        method: "historical", "parametric", "monte_carlo"
    
    Returns:
        float: VaR değeri (yüzde cinsinden)
    """
    ret = np.array(returns)
    
    if method == "historical":
        # Tarihsel VaR: belirtilen quantile
        return np.percentile(ret, (1 - confidence) * 100)
    
    elif method == "parametric":
        # Parametrik VaR: Normal dağılım varsayımı
        mean = np.mean(ret)
        std = np.std(ret)
        z_score = scipy_stats.norm.ppf(1 - confidence)
        return mean + z_score * std
    
    else:
        raise ValueError(f"Bilinmeyen VaR metodu: {method}")


def calculate_cvar(
    returns: Union[np.ndarray, List[float]], 
    confidence: float = 0.95
) -> float:
    """
    Conditional Value at Risk (CVaR / Expected Shortfall) hesaplar.
    
    CVaR = mean(returns <= VaR)
    
    Args:
        returns: Getiri dizisi
        confidence: Güven seviyesi (varsayılan: 0.95)
    
    Returns:
        float: CVaR değeri (yüzde cinsinden)
    """
    ret = np.array(returns)
    var = calculate_var(ret, confidence, method="historical")
    return np.mean(ret[ret <= var])


def calculate_sharpe(
    returns: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 1
) -> float:
    """
    Sharpe Oranı (bootstrap/proxy versiyonu).
    
    Sharpe = (mean - risk_free) / std
    
    Args:
        returns: Getiri dizisi
        risk_free_rate: Risksiz faiz oranı (varsayılan: 0)
        periods_per_year: Yıllık periyot sayısı (varsayılan: 1)
    
    Returns:
        float: Sharpe oranı
    """
    ret = np.array(returns)
    mean = np.mean(ret)
    std = np.std(ret)
    
    if std == 0:
        return np.nan
    
    sharpe = (mean - risk_free_rate) / std
    return sharpe * np.sqrt(periods_per_year)


def calculate_sortino(
    returns: Union[np.ndarray, List[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 1
) -> float:
    """
    Sortino Oranı (sadece negatif getirileri cezalandırır).
    
    Sortino = (mean - risk_free) / std(negative_returns)
    
    Args:
        returns: Getiri dizisi
        risk_free_rate: Risksiz faiz oranı (varsayılan: 0)
        periods_per_year: Yıllık periyot sayısı (varsayılan: 1)
    
    Returns:
        float: Sortino oranı
    """
    ret = np.array(returns)
    mean = np.mean(ret)
    negative = ret[ret < 0]
    
    if len(negative) == 0:
        return np.inf
    
    neg_std = np.std(negative)
    if neg_std == 0:
        return np.nan
    
    sortino = (mean - risk_free_rate) / neg_std
    return sortino * np.sqrt(periods_per_year)


def calculate_calmar(
    returns: Union[np.ndarray, List[float]],
    max_drawdown: Optional[float] = None
) -> float:
    """
    Calmar Oranı (proxy versiyonu).
    
    Calmar = mean_return / max_drawdown
    
    Args:
        returns: Getiri dizisi (max_drawdown hesaplanacaksa)
        max_drawdown: Önceden hesaplanmış max drawdown (opsiyonel)
    
    Returns:
        float: Calmar oranı (proxy)
    """
    ret = np.array(returns)
    mean = np.mean(ret)
    
    if max_drawdown is None:
        # Basit drawdown hesaplama
        cum = (1 + ret / 100).cumprod()
        peak = cum.cummax()
        dd = (cum / peak - 1) * 100
        max_dd = abs(np.min(dd))
    
    if max_dd == 0:
        return np.inf
    
    return mean / max_dd


def calculate_confidence_interval(
    data: Union[np.ndarray, List[float]],
    confidence: float = 0.95
) -> tuple:
    """
    Güven aralığı hesaplar.
    
    Args:
        data: Veri dizisi
        confidence: Güven seviyesi (varsayılan: 0.95)
    
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    arr = np.array(data)
    low = (1 - confidence) / 2
    high = 1 - low
    return (np.percentile(arr, low * 100), np.percentile(arr, high * 100))


def calculate_bootstrap_stats(
    bootstrap_results: Union[np.ndarray, List[float]],
    original: float,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Bootstrap sonuçlarından kapsamlı istatistikler.
    
    Args:
        bootstrap_results: Bootstrap simülasyon sonuçları
        original: Orijinal değer
        confidence: Güven seviyesi (varsayılan: 0.95)
    
    Returns:
        Dict: {
            'mean': float,
            'median': float,
            'std': float,
            'bias': float,
            'var_95': float,
            'cvar_95': float,
            'ci_lower': float,
            'ci_upper': float,
            'p_less_than_zero': float,
            'p_less_than_original': float,
            'sharpe': float,
            'sortino': float,
            'calmar': float,
            'skewness': float,
            'kurtosis': float,
            'min': float,
            'max': float,
        }
    """
    arr = np.array(bootstrap_results)
    
    # Temel metrikler
    mean = np.mean(arr)
    median = np.median(arr)
    std = np.std(arr)
    
    # Bias
    bias = mean - original
    
    # VaR ve CVaR
    var_95 = calculate_var(arr, confidence)
    cvar_95 = calculate_cvar(arr, confidence)
    
    # Güven aralığı
    ci_lower, ci_upper = calculate_confidence_interval(arr, confidence)
    
    # Olasılıklar
    p_less_than_zero = np.mean(arr < 0) * 100
    p_less_than_original = np.mean(arr < original) * 100
    
    # Risk-adjusted oranlar
    sharpe = calculate_sharpe(arr)
    sortino = calculate_sortino(arr)
    calmar = calculate_calmar(arr)
    
    # Dağılım şekli
    skewness = float(scipy_stats.skew(arr)) if len(arr) > 1 else 0.0
    kurtosis = float(scipy_stats.kurtosis(arr)) if len(arr) > 1 else 0.0
    
    return {
        "mean": mean,
        "median": median,
        "std": std,
        "bias": bias,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_less_than_zero": p_less_than_zero,
        "p_less_than_original": p_less_than_original,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "min": np.min(arr),
        "max": np.max(arr),
    }