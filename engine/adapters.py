# adapters.py
from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Dict, Optional

import pandas as pd

from engine.core import BacktestConfig

logger = logging.getLogger(__name__)

EXCEL_DOSYASI = "BIST_TUM_OHLCV_TL_USD_GUNCEL.xlsx"


class ExcelDataFetcher:
    """Excel'den veri çeker (tüm sayfalar bellekte)."""

    def __init__(self, cfg: BacktestConfig, excel_path: str):
        self.config = cfg
        self.excel_path = excel_path
        self._data: Dict[str, pd.DataFrame] = {}
        self._cache: Dict[tuple, pd.DataFrame] = {}
        self._lock = threading.Lock()
        self._load_all_sheets()

    def _load_all_sheets(self):
        try:
            xl = pd.ExcelFile(self.excel_path)
            for sheet in xl.sheet_names:
                if sheet == "OZET":
                    continue
                df = xl.parse(sheet, index_col=0, parse_dates=True)
                df.rename(
                    columns={
                        "Open_TL": "Open",
                        "High_TL": "High",
                        "Low_TL": "Low",
                        "Close_TL": "Close",
                    },
                    inplace=True,
                )
                df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
                self._data[sheet] = df
            logger.info(f"Excel yüklendi: {len(self._data)} hisse")
        except Exception as e:
            raise ValueError(f"Excel yükleme hatası: {str(e)}")

    def get_ohlcv(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        if start is None:
            start = self.config.data_start
        if end is None:
            end = self.config.backtest_end or datetime.now().strftime("%Y-%m-%d")

        cache_key = (symbol, start, end)
        with self._lock:
            if cache_key in self._cache:
                logger.debug(f"Cache hit: {symbol}")
                return self._cache[cache_key].copy()

        df = self._data.get(symbol)
        if df is None:
            raise ValueError(f"Symbol {symbol} not found in Excel.")

        start_ts = pd.Timestamp(start).normalize()
        end_ts = pd.Timestamp(end).normalize()
        df = df.loc[start_ts:end_ts].copy()

        if df.empty:
            raise ValueError(f"Veri gelmedi ({symbol}, {start}-{end})")
        if len(df) < self.config.min_data_count:
            raise ValueError(f"Yetersiz veri: {len(df)} satir ({symbol})")

        with self._lock:
            self._cache[cache_key] = df
        return df.copy()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            logger.info("Cache temizlendi")


try:
    import borsapy as bp
except ImportError:
    bp = None


class BorsapyDataFetcher:
    """Borsapy API'den veri çeker ve sembol/tarih bazında önbelleğe alır."""

    def __init__(self, cfg: BacktestConfig):
        if bp is None:
            raise ImportError("Borsapy sürümü için: pip install borsapy")
        self.config = cfg
        self._cache: Dict[tuple, pd.DataFrame] = {}
        self._lock = threading.Lock()

    def get_ohlcv(
        self, symbol: str, start: Optional[str] = None, end: Optional[str] = None
    ) -> pd.DataFrame:
        start = start or self.config.data_start
        end = end or self.config.backtest_end or datetime.now().strftime("%Y-%m-%d")
        cache_key = (symbol, start, end)

        with self._lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug(f"Cache hit: {symbol}")
            return cached.copy()

        try:
            df = bp.Ticker(symbol).history(start=start, end=end)
        except Exception as e:
            raise ValueError(f"Veri çekme hatası ({symbol}, {start}-{end}): {e}") from e

        if df is None or df.empty:
            raise ValueError(f"Veri gelmedi ({symbol}, {start}-{end})")

        gerekli = ["Open", "High", "Low", "Close", "Volume"]
        eksik = [c for c in gerekli if c not in df.columns]
        if eksik:
            raise ValueError(f"Eksik OHLCV sütunları ({symbol}): {eksik}")

        df = df[gerekli].copy()
        df.index = pd.to_datetime(df.index)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.astype(float)

        if len(df) < self.config.min_data_count:
            raise ValueError(f"Yetersiz veri: {len(df)} satır ({symbol})")

        with self._lock:
            self._cache[cache_key] = df
        return df.copy()

    def clear_cache(self):
        with self._lock:
            self._cache.clear()
            logger.info("Cache temizlendi")