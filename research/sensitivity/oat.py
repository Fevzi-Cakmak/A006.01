# research/sensitivity/oat.py
import numpy as np
import pandas as pd
from .base import SensitivityAnalyzer

class OAT(SensitivityAnalyzer):
    """One-at-a-Time duyarlılık analizi."""
    
    def run(self) -> pd.DataFrame:
        # (mevcut one_at_a_time mantığı)
        pass