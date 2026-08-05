from __future__ import annotations

import logging
from pathlib import Path
from typing import Set

import pandas as pd

from engine.core import _sembol_normalize

logger = logging.getLogger(__name__)


def dosya_bul(klasor: Path, desen: str) -> Path:
    """
    Verilen klasörde desene uyan en son güncellenen dosyayı döndürür.
    """

    bulunanlar = sorted(
        klasor.glob(desen),
        key=lambda p: p.stat().st_mtime,
    )

    if not bulunanlar:
        raise FileNotFoundError(f"Dosya bulunamadı: {desen}")

    if len(bulunanlar) > 1:
        logger.warning(
            "Birden fazla dosya bulundu. "
            "En son güncellenen dosya kullanılacak: %s",
            bulunanlar[-1],
        )

    return bulunanlar[-1]


def ozetten_kodlari_oku(excel_yolu: str) -> Set[str]:
    """
    Excel dosyasındaki OZET sayfasından aktif hisse kodlarını okur.
    """

    df = pd.read_excel(excel_yolu, sheet_name="OZET")

    if df.empty:
        raise ValueError(f"OZET sayfası boş: {excel_yolu}")

    kolon_haritasi = {
        str(col).strip().casefold(): col
        for col in df.columns
    }

    hisse_kolonu = next(
        (
            kolon_haritasi[x]
            for x in ("hisse", "sembol", "symbol", "kod", "ticker")
            if x in kolon_haritasi
        ),
        None,
    )

    if hisse_kolonu is None:
        raise ValueError(
            f"OZET sayfasında Hisse sütunu bulunamadı: {excel_yolu}"
        )

    calisma = df.copy()

    durum_kolonu = kolon_haritasi.get("durum")

    if durum_kolonu is not None:
        durum = (
            calisma[durum_kolonu]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        calisma = calisma[
            durum.isin({"OK", "AKTIF", "AKTİF", "TRUE", "1"})
        ]

    kodlar = {
        _sembol_normalize(v)
        for v in calisma[hisse_kolonu]
        if _sembol_normalize(v)
    }

    if not kodlar:
        raise ValueError(
            f"OZET sayfasından hisse kodu okunamadı: {excel_yolu}"
        )

    return kodlar


def _log_missing(message: str, symbols: Set[str]) -> None:
    """
    Eksik sembolleri okunabilir şekilde loglar.

    İlk 10 sembol gösterilir.
    Daha fazla sembol varsa kalan adet belirtilir.
    """

    symbols_sorted = sorted(symbols)

    shown = symbols_sorted[:10]

    preview = ", ".join(shown)

    if len(symbols_sorted) > 10:
        preview += f", ... (+{len(symbols_sorted) - 10} adet daha)"

    logger.warning(
        "%s (%d adet): %s",
        message,
        len(symbols_sorted),
        preview,
    )


def endeks_listelerini_dogrula(
    kodlar: Set[str],
    b30: Set[str],
    b50: Set[str],
    b100: Set[str],
) -> None:
    """
    Endeks listelerinin tutarlılığını kontrol eder.

    Beklenen ilişki:

        BIST30 ⊂ BIST50 ⊂ BIST100 ⊂ BIST TÜM
    """

    if not b30:
        raise ValueError("BIST30 listesi boş.")

    if not b50:
        raise ValueError("BIST50 listesi boş.")

    if not b100:
        raise ValueError("BIST100 listesi boş.")

    # ---------------------------------------------------------
    # BIST50, BIST30'u kapsamalıdır.
    # ---------------------------------------------------------
    eksik_b50 = b30 - b50

    if eksik_b50:
        _log_missing(
            "Aşağıdaki BIST30 hisseleri BIST50 listesinde bulunmuyor",
            eksik_b50,
        )

    # ---------------------------------------------------------
    # BIST100, BIST50'yi kapsamalıdır.
    # ---------------------------------------------------------
    eksik_b100 = b50 - b100

    if eksik_b100:
        _log_missing(
            "Aşağıdaki BIST50 hisseleri BIST100 listesinde bulunmuyor",
            eksik_b100,
        )

    # ---------------------------------------------------------
    # Endekslerde olup TÜM evreninde olmayan hisseler.
    # ---------------------------------------------------------
    bilinmeyen = (b30 | b50 | b100) - kodlar

    if bilinmeyen:
        _log_missing(
            "BIST TÜM evreninde bulunmayan endeks hisseleri",
            bilinmeyen,
        )