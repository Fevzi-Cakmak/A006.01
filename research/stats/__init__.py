# research/stats/__init__.py
"""
İstatistiksel Metrikler Modülü

- Bias: Ortalama - Orijinal
- VaR: Value at Risk
- CVaR: Conditional Value at Risk (Expected Shortfall)
- Sharpe, Sortino, Calmar: Risk-adjusted oranlar
- Bootstrap Stats: Bootstrap dağılımından metrikler
"""

from .stats import (
    calculate_bias,
    calculate_var,
    calculate_cvar,
    calculate_sharpe,
    calculate_sortino,
    calculate_calmar,
    calculate_bootstrap_stats,
    calculate_confidence_interval,
    StatResult,
)

__all__ = [
    "calculate_bias",
    "calculate_var",
    "calculate_cvar",
    "calculate_sharpe",
    "calculate_sortino",
    "calculate_calmar",
    "calculate_bootstrap_stats",
    "calculate_confidence_interval",
    "StatResult",
]