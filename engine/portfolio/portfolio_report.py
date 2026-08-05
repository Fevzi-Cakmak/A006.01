from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import numpy as np

class PortfolioReport:
    def __init__(self, cfg):
        self.config = cfg
        self.gunluk: List[dict] = []

    # ... (diğer metodlar)