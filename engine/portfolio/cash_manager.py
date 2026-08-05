from __future__ import annotations

class CashManager:
    def __init__(self, cfg):
        self.config = cfg
        self.nakit = cfg.initial_capital

    # ... (diğer metodlar)