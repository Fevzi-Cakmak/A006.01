# runner.py
from __future__ import annotations

import logging
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Iterable, Set

import pandas as pd

from engine import core as engine_core  # engine.core'u import et
from engine.core import (
    COLAB,
    ExcelReporter,
    DataFetcherProtocol,
    BacktestConfig,
    ayarlar_hazirla,
    backtest_hisse,
    cikis_analizi_hazirla,
    endeks_ozeti_hazirla,
    endeks_uyeligi,
    faktor_analizi_hazirla,
    genel_ozet_hazirla,
    hisse_ozeti_hazirla,
    sim001_excel_ekle,
    skor_analizi_hazirla,
    yillik_ozet_hazirla,
)

logger = logging.getLogger(__name__)

_RENAME = {
    "Giris Tarihi": "Giriş Tarihi",
    "Cikis Tarihi": "Çıkış Tarihi",
    "Giris Fiyati": "Giriş Fiyatı",
    "Cikis Fiyati": "Çıkış Fiyatı",
    "Giris Skoru": "Giriş Skoru",
    "Giris Sinyali": "Giriş Sinyali",
    "Giris RSI": "Giriş RSI",
    "Giris ADX": "Giriş ADX",
    "Giris Hacim Orani": "Giriş Hacim Oranı",
    "Giris MACD": "Giriş MACD",
    "Giris MACD Sinyal": "Giriş MACD Sinyal",
    "Giris ATR %": "Giriş ATR %",
    "BB Ust Banda Yakin": "BB Üst Banda Yakın",
    "Cikis Nedeni": "Çıkış Nedeni",
    "Bekleme Gunu": "Bekleme Günü",
    "Brut Getiri %": "Brüt Getiri %",
}


def run_backtest(
    data_fetcher: DataFetcherProtocol,
    kodlar: Iterable[str],
    b100: Set[str],
    b50: Set[str],
    b30: Set[str],
    source_name: str,
    filename_prefix: str,
    max_workers: int,
    cfg: BacktestConfig,
) -> str:
    kodlar = sorted(set(kodlar))
    worker_sayisi = min(
        max_workers,
        max(1, os.cpu_count() or 1),
        len(kodlar),
    )

    logger.info("=" * 68)
    logger.info(f"   VIOS A003 - {source_name} (A002 ALGORİTMASI / MODÜLER)")
    logger.info("=" * 68)
    logger.info(
        f"Skor:{cfg.strong_buy_min_score} | Hacim:{cfg.volume_high_ratio}-{cfg.volume_upper_ratio}x | ADX:{cfg.adx_strong}-{cfg.adx_upper} | RSI:{cfg.rsi_low}-{cfg.rsi_high}"
    )
    logger.info(
        f"+DI>-DI: Zorunlu | SMA200 Eğim: Pozitif zorunlu | Stop:%{cfg.stop_loss_ratio*100:.0f}"
    )
    logger.info(f"Hisse sayısı: {len(kodlar)} | Worker: {worker_sayisi}")

    tum, hata = [], []
    hata_sayisi = sinyal = islenen = 0
    hata_tipleri = Counter()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=worker_sayisi) as executor:
        future_to_kod = {
            executor.submit(
                backtest_hisse,
                kod,
                data_fetcher,
                cfg,
                cfg.backtest_start,
                cfg.backtest_end,
            ): kod
            for kod in kodlar
        }
        for future in as_completed(future_to_kod):
            kod = future_to_kod[future]
            islenen += 1
            try:
                s = future.result()
                if not s.empty:
                    s["Endeks Üyeliği"] = endeks_uyeligi(
                        kod, b100, b50, b30, set(kodlar)
                    )
                    tum.append(s)
                    sinyal += len(s)
            except Exception as e:
                hata_adi = type(e).__name__
                hata_tipleri[hata_adi] += 1
                hata.append({"Hisse": kod, "Hata": f"{kod}: {e}"})
                hata_sayisi += 1
                if hata_sayisi <= 5:
                    logger.warning(f"HATA ({kod}): {e}")
            if islenen % 50 == 0 or islenen == len(kodlar):
                logger.info(
                    f"  {islenen}/{len(kodlar)} ({time.time()-t0:.0f}s) | İşlem:{sinyal} | Hata:{hata_sayisi} | {kod}"
                )

    if hata_tipleri:
        logger.info("Hata dağılımı:")
        for hata_tipi, adet in hata_tipleri.most_common():
            logger.info(f"  {hata_tipi}: {adet}")

    if not tum:
        raise RuntimeError(
            f"Hiç işlem oluşmadı. "
            f"İşlenen hisse: {islenen}, "
            f"hatalı hisse: {hata_sayisi}, "
            f"başlangıç: {cfg.backtest_start}, "
            f"bitiş: {cfg.backtest_end or 'güncel'}"
        )

    islemler = pd.concat(tum, ignore_index=True).rename(columns=_RENAME)
    islemler = islemler.sort_values(["Giriş Tarihi", "Hisse"])
    bitis_tarihi = pd.to_datetime(islemler["Çıkış Tarihi"]).max().strftime("%Y-%m-%d")

    hisse = hisse_ozeti_hazirla(islemler)
    sheets = [
        (genel_ozet_hazirla(islemler, bitis_tarihi, cfg), "Genel Özet"),
        (endeks_ozeti_hazirla(islemler), "Endeks Özeti"),
        (hisse, "Hisse Özeti"),
        (islemler, "Tüm İşlemler"),
        (yillik_ozet_hazirla(islemler), "Yıllık Özet"),
        (skor_analizi_hazirla(islemler), "Skor Analizi"),
        (cikis_analizi_hazirla(islemler), "Çıkış Analizi"),
        (faktor_analizi_hazirla(islemler), "Faktör Analizi"),
        (hisse.head(25), "En İyi 25"),
        (
            hisse.sort_values("Bileşik Net Getiri %", ascending=True).head(25),
            "En Kötü 25",
        ),
        (pd.DataFrame(hata, columns=["Hisse", "Hata"]), "Veri Hataları"),
        (ayarlar_hazirla(cfg), "Ayarlar"),
    ]

    os.makedirs(cfg.output_dir, exist_ok=True)
    dosya = os.path.join(
        cfg.output_dir,
        f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
    )
    reporter = ExcelReporter(dosya)
    for df, name in sheets:
        reporter.add_sheet(df, name)
    reporter.save()

    sim001_excel_ekle(dosya, islemler, data_fetcher, cfg)

    logger.info(f"\n12+ sekmeli Excel hazır: {dosya}")
    if COLAB:
        engine_core.files.download(dosya)  # engine.core üzerinden eriş
    return dosya