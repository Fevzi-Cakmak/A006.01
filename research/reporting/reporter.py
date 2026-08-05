import os
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from ..constants import COLUMN_NAME_MAP

logger = logging.getLogger(__name__)

class ResearchReporter:
    """Tüm analiz sonuçlarını Excel dosyasına yazar."""
    
    def __init__(self, config, results_df: pd.DataFrame):
        self.config = config
        self.results_df = results_df
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excel_path = os.path.join(
            self.config.output_dir,
            f"research_summary_{self.timestamp}.xlsx"
        )
    
    def save_all(self, analysis_results: Dict[str, pd.DataFrame]):
        """Tüm sayfaları tek seferde yazar."""
        with pd.ExcelWriter(self.excel_path, engine='openpyxl') as writer:
            # 1. Ham sonuçlar
            df_main = self.results_df.copy()
            df_main.rename(columns=COLUMN_NAME_MAP, inplace=True)
            df_main.to_excel(writer, sheet_name='Tüm Sonuçlar', index=False)
            
            self._write_summary(writer)
            self._write_correlations(writer)
            
            for sheet_name, df in analysis_results.items():
                if df is not None and not df.empty:
                    if sheet_name in ('Pareto Özeti', 'Executive Summary'):
                        df.rename(columns=COLUMN_NAME_MAP, inplace=True, errors='ignore')
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        from ..plotting import add_plots_to_excel
        add_plots_to_excel(self.excel_path, self.results_df, self.config)
        
        logger.info(f"📊 Rapor kaydedildi: {self.excel_path}")
        return self.excel_path
    
    def _write_summary(self, writer):
        numeric_cols = self.results_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return
        summary = self.results_df[numeric_cols].describe().T
        summary['median'] = self.results_df[numeric_cols].median()
        summary['range'] = summary['max'] - summary['min']
        summary['cv'] = summary['std'] / summary['mean']
        summary['skewness'] = self.results_df[numeric_cols].skew()
        summary['kurtosis'] = self.results_df[numeric_cols].kurtosis()
        summary.index.rename('Parametre', inplace=True)
        summary.to_excel(writer, sheet_name='Özet İstatistikler')
    
    def _write_correlations(self, writer):
        param_cols = list(self.config.param_space.keys())
        metric_cols = ['compound_return', 'win_rate', 'profit_factor', 
                       'total_trades', 'max_win', 'max_loss', 'avg_return']
        available_metrics = [c for c in metric_cols if c in self.results_df.columns]
        corr_cols = param_cols + available_metrics
        if len(corr_cols) < 2:
            return
        turkish_corr_cols = [COLUMN_NAME_MAP.get(c, c) for c in corr_cols]
        
        for method, sheet_name in [('pearson', 'Korelasyon_Pearson'),
                                   ('spearman', 'Korelasyon_Spearman'),
                                   ('kendall', 'Korelasyon_Kendall')]:
            corr = self.results_df[corr_cols].corr(method=method)
            corr.columns = turkish_corr_cols
            corr.index = turkish_corr_cols
            corr.to_excel(writer, sheet_name=sheet_name, index=True)