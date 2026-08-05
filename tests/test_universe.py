import pytest
from pathlib import Path
from engine.universe import dosya_bul


def test_dosya_bul_not_found(tmp_path):
    """Dosya bulunamadığında FileNotFoundError fırlatılmalı."""
    with pytest.raises(FileNotFoundError):
        dosya_bul(tmp_path, "olmayan_dosya*.xlsx")


def test_dosya_bul_single_file(tmp_path):
    """Tek dosya bulunduğunda doğru dosya dönmeli."""
    test_file = tmp_path / "BIST_TUM_OHLCV_TL_USD_GUNCEL_20250101.xlsx"
    test_file.touch()
    
    result = dosya_bul(tmp_path, "BIST_TUM_OHLCV_TL_USD_GUNCEL*.xlsx")
    assert result == test_file


def test_dosya_bul_multiple_files(tmp_path):
    """Birden fazla dosya olduğunda en güncel (sondaki tarih) seçilmeli."""
    files = [
        "BIST_TUM_OHLCV_TL_USD_GUNCEL_20250101.xlsx",
        "BIST_TUM_OHLCV_TL_USD_GUNCEL_20250115.xlsx",
        "BIST_TUM_OHLCV_TL_USD_GUNCEL_20250120.xlsx",
    ]
    for fname in files:
        (tmp_path / fname).touch()
    
    result = dosya_bul(tmp_path, "BIST_TUM_OHLCV_TL_USD_GUNCEL*.xlsx")
    expected = tmp_path / "BIST_TUM_OHLCV_TL_USD_GUNCEL_20250120.xlsx"
    assert result == expected


def test_dosya_bul_with_pattern(tmp_path):
    """Desen farklı olduğunda da doğru seçim yapılmalı."""
    files = [
        "BIST_30_OHLCV_TL_USD_GUNCEL_20250101.xlsx",
        "BIST_30_OHLCV_TL_USD_GUNCEL_20250115.xlsx",
    ]
    for fname in files:
        (tmp_path / fname).touch()
    
    result = dosya_bul(tmp_path, "BIST_30_OHLCV_TL_USD_GUNCEL*.xlsx")
    expected = tmp_path / "BIST_30_OHLCV_TL_USD_GUNCEL_20250115.xlsx"
    assert result == expected