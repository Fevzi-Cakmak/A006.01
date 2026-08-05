from __future__ import annotations

from typing import Any

import pandas as pd

__all__ = ["generate_executive_summary"]

EPSILON: float = 1e-6

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "compound_return",
        "profit_factor",
        "max_loss",
        "total_trades",
        "stop_loss_ratio",
        "trailing_stop_ratio",
        "rsi_low",
        "adx_strong",
        "volume_high_ratio",
        "win_rate",
    }
)


def generate_executive_summary(
    df: pd.DataFrame,
    min_trades_for_best_pf: int = 30,
    min_trades_for_balanced: int = 40,
    max_loss_threshold: float = -14.0,
) -> pd.DataFrame:
    """
    Research sonuçlarından Executive Summary oluşturur.

    Oluşturulan senaryolar:

    - Max Return
    - Best PF
    - Lowest Risk
    - Balanced
    - Most Robust

    Parameters
    ----------
    df
        Research sonuçlarını içeren DataFrame.

    Returns
    -------
    pd.DataFrame
        Executive Summary tablosu.
    """

    if df.empty:
        return pd.DataFrame()

    missing = REQUIRED_COLUMNS - set(df.columns)

    if missing:
        raise ValueError(
            "Eksik sütunlar: "
            + ", ".join(sorted(missing))
        )

    summaries: list[dict[str, Any]] = []

    # ---------------------------------------------------------
    # 1. Max Return
    # ---------------------------------------------------------
    row = df.loc[df["compound_return"].idxmax()]
    summaries.append(_make_row("Max Return", row))

    # ---------------------------------------------------------
    # 2. Best Profit Factor
    # ---------------------------------------------------------
    filtered = df[
        df["total_trades"] >= min_trades_for_best_pf
    ]

    if not filtered.empty:
        row = filtered.loc[
            filtered["profit_factor"].idxmax()
        ]
        summaries.append(_make_row("Best PF", row))

    # ---------------------------------------------------------
    # 3. Lowest Risk
    # ---------------------------------------------------------
    row = df.loc[df["max_loss"].idxmax()]
    summaries.append(_make_row("Lowest Risk", row))

    # ---------------------------------------------------------
    # 4. Balanced
    # ---------------------------------------------------------
    filtered = df[
        (df["total_trades"] >= min_trades_for_balanced)
        & (df["max_loss"] > max_loss_threshold)
    ]

    if not filtered.empty:
        row = filtered.loc[
            filtered["profit_factor"].idxmax()
        ]
        summaries.append(_make_row("Balanced", row))

    # ---------------------------------------------------------
    # 5. Most Robust
    # ---------------------------------------------------------
    df_copy = df.copy()

    df_copy["return_risk_ratio"] = (
        df_copy["compound_return"]
        / (-df_copy["max_loss"] + EPSILON)
    )

    row = df_copy.loc[
        df_copy["return_risk_ratio"].idxmax()
    ]

    summaries.append(_make_row("Most Robust", row))

    return pd.DataFrame(summaries)


def _make_row(
    scenario: str,
    row: pd.Series,
) -> dict[str, Any]:
    """
    Executive Summary için tek satır oluşturur.
    """

    return {
        "Senaryo": scenario,
        "stop_loss": row["stop_loss_ratio"],
        "trailing": row["trailing_stop_ratio"],
        "rsi": row["rsi_low"],
        "adx": row["adx_strong"],
        "volume": row["volume_high_ratio"],
        "compound_return": row["compound_return"],
        "profit_factor": row["profit_factor"],
        "win_rate": row["win_rate"],
        "max_loss": row["max_loss"],
        "total_trades": row["total_trades"],
    }