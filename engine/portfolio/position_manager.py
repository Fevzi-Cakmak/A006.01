from __future__ import annotations
from typing import Dict, List
import pandas as pd

class PositionManager:
    """Açık pozisyonları ve işlem giriş/çıkışlarını yönetir."""

    def __init__(self, cfg):
        self.config = cfg
        self.acik: Dict[str, dict] = {}
        self.kapali: List[dict] = []
        self.ek_alimlar: List[dict] = []
        self.atlanan: List[dict] = []

    # ... (diğer metodlar aynen, import yok)