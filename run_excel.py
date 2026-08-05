# run_excel.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from engine.adapters import ExcelDataFetcher
from engine.core import config
from engine.runner import run_backtest
from engine.universe import dosya_bul, ozetten_kodlari_oku, endeks_listelerini_dogrula


def main() -> None:
    p = argparse.ArgumentParser(
        description="VIOS A003 Excel veri kaynağı - otomatik dosya bulma"
    )
    p.add_argument(
        "--excel",
        default="BIST_TUM_OHLCV_TL_USD_GUNCEL*.xlsx",
        help="BIST TÜM dosya deseni",
    )
    p.add_argument(
        "--bist30",
        default="BIST_30_OHLCV_TL_USD_GUNCEL*.xlsx",
        help="BIST 30 dosya deseni",
    )
    p.add_argument(
        "--bist50",
        default="BIST_50_OHLCV_TL_USD_GUNCEL*.xlsx",
        help="BIST 50 dosya deseni",
    )
    p.add_argument(
        "--bist100",
        default="BIST_100_OHLCV_TL_USD_GUNCEL*.xlsx",
        help="BIST 100 dosya deseni",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Paralel çalışan sayısı (otomatik sınırlanır)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log seviyesi",
    )
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    klasor = Path(".")
    tum_dosyasi = dosya_bul(klasor, args.excel)
    b30_dosyasi = dosya_bul(klasor, args.bist30)
    b50_dosyasi = dosya_bul(klasor, args.bist50)
    b100_dosyasi = dosya_bul(klasor, args.bist100)

    kodlar = sorted(ozetten_kodlari_oku(tum_dosyasi))
    b30 = ozetten_kodlari_oku(b30_dosyasi)
    b50_tam = ozetten_kodlari_oku(b50_dosyasi)
    b100_tam = ozetten_kodlari_oku(b100_dosyasi)

    b50 = b50_tam - b30
    b100 = b100_tam - b50_tam

    endeks_listelerini_dogrula(set(kodlar), b30, b50, b100)

    print(
        "Endeks üyelikleri hazır | "
        f"BIST30:{len(b30)} | "
        f"BIST50 ilave:{len(b50)} (toplam {len(b50_tam)}) | "
        f"BIST100 ilave:{len(b100)} (toplam {len(b100_tam)}) | "
        f"BIST TÜM:{len(kodlar)}"
    )

    for kod in ("BIMAS", "VAKBN", "CIMSA", "PASEU", "BRYAT", "MAGEN", "VESTL"):
        if kod in kodlar:
            uyelik = (
                "BIST30"
                if kod in b30
                else ("BIST50" if kod in b50 else ("BIST100" if kod in b100 else "BIST TÜM"))
            )
            print(f"  {kod}: {uyelik}")

    fetcher = ExcelDataFetcher(config, str(tum_dosyasi))
    run_backtest(
        fetcher,
        kodlar,
        b100,
        b50,
        b30,
        "EXCEL VERİ KAYNAĞI",
        "A003_EXCEL",
        args.workers,
        config,
    )


if __name__ == "__main__":
    main()