#!/usr/bin/env python3
"""
VIOS A006 – Quant Research Runner
"""

import logging
from pathlib import Path

from engine.adapters import ExcelDataFetcher
from engine.core import config as default_config
from engine.universe import (
    dosya_bul,
    ozetten_kodlari_oku,
    endeks_listelerini_dogrula,
)

from research import (
    ResearchConfig,
    ResearchRunner,
)


def main():
    # =============================================================
    # 1. LOGGING YAPILANDIRMASI (Tek ve Doğru Yer) - Öneri #2
    # =============================================================
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # =============================================================

    root = Path(".")

    tum = dosya_bul(root, "BIST_TUM_OHLCV_TL_USD_GUNCEL*.xlsx")
    b30_file = dosya_bul(root, "BIST_30_OHLCV_TL_USD_GUNCEL*.xlsx")
    b50_file = dosya_bul(root, "BIST_50_OHLCV_TL_USD_GUNCEL*.xlsx")
    b100_file = dosya_bul(root, "BIST_100_OHLCV_TL_USD_GUNCEL*.xlsx")

    symbols = sorted(ozetten_kodlari_oku(tum))

    b30 = ozetten_kodlari_oku(b30_file)
    b50_all = ozetten_kodlari_oku(b50_file)
    b100_all = ozetten_kodlari_oku(b100_file)

    b50 = b50_all - b30
    b100 = b100_all - b50_all

    endeks_listelerini_dogrula(
        set(symbols),
        b30,
        b50,
        b100,
    )

    fetcher = ExcelDataFetcher(
        default_config,
        str(tum),
    )

    # =============================================================
    # 2. KONFIGÜRASYON (ASCII Uyumlu ve Geçerli Parametreler)
    # =============================================================
    cfg = ResearchConfig(
        sampling_method="random",
        num_samples=50,
        random_seed=42,
        parallel_workers=1,
        # *** ÖNEMLİ: ASCII uyumlu hedef değişken adı (Öneri #1) ***
        target_metric="compound_return",
        output_dir="research_outputs",
        save_individual_reports=False,  # Geçerli parametre

        # (İsteğe bağlı) Yeni analiz parametreleri - varsayılanlar zaten tanımlı,
        # isterseniz burada ezebilirsiniz:
        # rf_n_estimators=500,
        # bootstrap_n_iterations=1000,
        # pareto_top_n=20,
        # min_trades_for_balanced=40,

        param_space={
            "stop_loss_ratio": {
                "low": 0.08,
                "high": 0.12,
                "dist": "uniform",
            },
            "trailing_stop_ratio": {
                "low": 0.05,
                "high": 0.10,
                "dist": "uniform",
            },
            "rsi_low": {
                "low": 55,
                "high": 65,
                "dist": "uniform",
                "dtype": "int",
            },
            "adx_strong": {
                "low": 28,
                "high": 35,
                "dist": "uniform",
                "dtype": "int",
            },
            "volume_high_ratio": {
                "low": 1.8,
                "high": 2.5,
                "dist": "uniform",
            },
        },
    )
    # =============================================================

    runner = ResearchRunner(
        config=cfg,
        data_fetcher=fetcher,
        symbols=symbols,
        b30=b30,
        b50=b50,
        b100=b100,
    )

    runner.run()


if __name__ == "__main__":
    main()