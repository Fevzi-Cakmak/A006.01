# research/stats/test_stats.py
import numpy as np
import pytest
from .stats import (
    calculate_bias,
    calculate_var,
    calculate_cvar,
    calculate_sharpe,
    calculate_sortino,
    calculate_calmar,
    calculate_bootstrap_stats,
    calculate_confidence_interval,
)


def test_calculate_bias():
    simulated = [10, 20, 30, 40, 50]
    original = 25
    assert calculate_bias(simulated, original) == 5.0

    simulated = [10, 20, 30]
    original = 30
    assert calculate_bias(simulated, original) == -10.0


def test_calculate_var():
    returns = [-5, -3, -1, 0, 1, 2, 3, 4, 5, 10]
    var_95 = calculate_var(returns, 0.95)
    # %5 quantile -> -3
    assert abs(var_95 - (-3)) < 0.01


def test_calculate_cvar():
    returns = [-10, -5, -3, -1, 0, 1, 2, 3, 4, 5]
    cvar_95 = calculate_cvar(returns, 0.95)
    # VaR = -3, CVaR = mean of <= -3 = (-10-5-3)/3 = -6
    assert abs(cvar_95 - (-6)) < 0.01


def test_calculate_sharpe():
    returns = [10, 5, 8, 12, 6]
    sharpe = calculate_sharpe(returns, risk_free_rate=0)
    assert sharpe > 0


def test_calculate_sortino():
    returns = [10, -5, 8, -3, 12]
    sortino = calculate_sortino(returns, risk_free_rate=0)
    assert sortino > 0


def test_calculate_calmar():
    returns = [10, 5, -2, 8, 3]
    calmar = calculate_calmar(returns)
    assert calmar > 0


def test_calculate_confidence_interval():
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    low, high = calculate_confidence_interval(data, 0.95)
    # %2.5 ve %97.5 quantile
    assert low == 1.25
    assert high == 9.75


def test_calculate_bootstrap_stats():
    bootstrap_results = [
        1000, 1100, 1200, 900, 950,
        1050, 1150, 1250, 850, 1080
    ]
    original = 1000
    stats = calculate_bootstrap_stats(bootstrap_results, original, 0.95)
    
    assert "mean" in stats
    assert "bias" in stats
    assert "var_95" in stats
    assert "cvar_95" in stats
    assert "sharpe" in stats
    assert "sortino" in stats
    assert "calmar" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats
    assert "p_less_than_zero" in stats
    assert "p_less_than_original" in stats