import os
import sys
import json
import copy
import logging
from datetime import datetime
from typing import Dict, Any, List, Set

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config.research_config import ResearchConfig
from .sampling import RandomSampler, LatinHypercubeSampler, GridSampler
from .reporting.reporter import ResearchReporter

# Yeni analiz modülleri
try:
    from .importance.random_forest import compute_rf_importance
    from .importance.permutation import compute_permutation_importance
    from .stats.significance import batch_correlation_analysis
    from .reporting.pareto import compute_multi_criteria_pareto
    from .reporting.executive_summary import generate_executive_summary
    NEW_ANALYSIS_AVAILABLE = True
except ImportError as e:
    NEW_ANALYSIS_AVAILABLE = False
    logging.warning(f"Yeni analiz modülleri yüklenemedi: {e}")

# Engine importları
from engine.core import BacktestConfig, DataFetcherProtocol, config as default_config
from engine.runner import run_backtest

# Logger tanımı (basicConfig dışarıda yapılacak)
logger = logging.getLogger(__name__)


class ResearchRunner:
    def __init__(self, config: ResearchConfig, data_fetcher, symbols, b30, b50, b100):
        self.config = config
        self.data_fetcher = data_fetcher
        self.symbols = symbols
        self.b30 = b30
        self.b50 = b50
        self.b100 = b100
        self.results_df = pd.DataFrame()
        self.best_params = {}
        os.makedirs(self.config.output_dir, exist_ok=True)
        # logging.basicConfig buradan KALDIRILDI (dışarıda tek sefer yapılacak)

    def _get_sampler(self):
        if self.config.sampling_method == "random":
            return RandomSampler(self.config.param_space, self.config.random_seed)
        elif self.config.sampling_method == "latin_hypercube":
            return LatinHypercubeSampler(self.config.param_space, self.config.random_seed)
        elif self.config.sampling_method == "grid":
            return GridSampler(self.config.param_space)
        else:
            raise ValueError(f"Bilinmeyen örnekleme metodu: {self.config.sampling_method}")

    def _build_backtest_config(self, params):
        cfg = copy.deepcopy(default_config)
        for key, value in params.items():
            parts = key.split('.')
            if len(parts) == 1 and hasattr(cfg, key):
                setattr(cfg, key, value)
            elif len(parts) == 2:
                parent, child = parts
                if hasattr(cfg, parent) and hasattr(getattr(cfg, parent), child):
                    setattr(getattr(cfg, parent), child, value)
        return cfg

    def _run_single_backtest(self, params):
        try:
            cfg = self._build_backtest_config(params)
            result_file = run_backtest(
                data_fetcher=self.data_fetcher,
                kodlar=self.symbols,
                b100=self.b100,
                b50=self.b50,
                b30=self.b30,
                source_name="Araştırma",
                filename_prefix="RESEARCH",
                max_workers=1,
                cfg=cfg
            )
            # Metrikleri oku (ASCII isimlere dönüştür)
            df = pd.read_excel(result_file, sheet_name="Genel Özet")
            metrics = {}
            for _, row in df.iterrows():
                if row["Açıklama"] == "Teorik Bileşik Getiri %":
                    metrics["compound_return"] = float(row["Değer"])
                elif row["Açıklama"] == "Başarı Oranı %":
                    metrics["win_rate"] = float(row["Değer"])
                elif row["Açıklama"] == "Toplam İşlem":
                    metrics["total_trades"] = int(row["Değer"])
                elif row["Açıklama"] == "Profit Factor":
                    metrics["profit_factor"] = float(row["Değer"]) if row["Değer"] != "∞" else 999.0
                elif row["Açıklama"] == "En İyi İşlem %":
                    metrics["max_win"] = float(row["Değer"])
                elif row["Açıklama"] == "En Kötü İşlem %":
                    metrics["max_loss"] = float(row["Değer"])
                elif row["Açıklama"] == "Ortalama İşlem %":
                    metrics["avg_return"] = float(row["Değer"])

            if not self.config.save_individual_reports:
                os.remove(result_file)

            return {**params, **metrics, 'status': 'SUCCESS'}
        except Exception as e:
            logger.error(f"Backtest başarısız: {e}")
            return {**params, 'status': 'FAILED', 'error': str(e)}

    def _run_parallel(self, param_list):
        results = []
        if self.config.parallel_workers > 1:
            with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
                futures = {executor.submit(self._run_single_backtest, p): p for p in param_list}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Backtest ilerlemesi"):
                    results.append(future.result())
        else:
            for params in tqdm(param_list, desc="Backtest ilerlemesi"):
                results.append(self._run_single_backtest(params))
        return results

    def run(self):
        logger.info(f"Araştırma başlatılıyor: {self.config.sampling_method} ({self.config.num_samples} örnek)")
        sampler = self._get_sampler()
        param_samples = sampler.sample(self.config.num_samples)
        raw_results = self._run_parallel(param_samples)
        self.results_df = pd.DataFrame(raw_results)
        self.results_df = self.results_df[self.results_df['status'] == 'SUCCESS']
        if self.results_df.empty:
            logger.error("Başarılı hiçbir backtest sonucu yok!")
            return
        target = self.config.target_metric
        if target in self.results_df.columns:
            best_idx = self.results_df[target].idxmax()
            self.best_params = self.results_df.loc[best_idx].to_dict()
            logger.info(f"En iyi {target}: {self.results_df.loc[best_idx][target]:.2f}")
        
        # Analizleri çalıştır
        analysis_results = self._run_analyses()
        
        # Raporu oluştur
        reporter = ResearchReporter(self.config, self.results_df)
        excel_path = reporter.save_all(analysis_results)
        
        # En iyi parametreleri JSON olarak kaydet
        best_path = os.path.join(self.config.output_dir, f"best_params_{reporter.timestamp}.json")
        with open(best_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        logger.info(f"En iyi parametreler: {best_path}")

    def _run_analyses(self) -> Dict[str, pd.DataFrame]:
        """Tüm gelişmiş analizleri çalıştırır ve DataFrame'leri döndürür."""
        if not NEW_ANALYSIS_AVAILABLE:
            logger.warning("Yeni analiz modülleri mevcut değil.")
            return {}
        
        results = {}
        param_cols = list(self.config.param_space.keys())
        target = self.config.target_metric  # 'compound_return'
        
        if len(param_cols) < 2 or target not in self.results_df.columns:
            logger.warning("Yetersiz veri, analizler atlanıyor.")
            return results
        
        # 1. Feature Importance (RF MDI + Permutation + Ortak Skor)
        try:
            logger.info("Feature Importance hesaplanıyor...")
            rf_imp = compute_rf_importance(
                self.results_df, target, param_cols,
                n_estimators=self.config.rf_n_estimators,
                random_state=self.config.rf_random_state
            )
            perm_imp = compute_permutation_importance(
                self.results_df, target, param_cols,
                n_repeats=self.config.permutation_n_repeats,
                n_estimators=self.config.rf_n_estimators,
                random_state=self.config.rf_random_state
            )
            combined = pd.merge(rf_imp, perm_imp, on='feature', how='outer')
            
            # Normalize et (0-1 arası) ve ortak skor oluştur
            combined['rf_norm'] = combined['rf_mdi_importance'] / combined['rf_mdi_importance'].max()
            combined['perm_norm'] = combined['permutation_mean'] / combined['permutation_mean'].max()
            combined['combined_score'] = (combined['rf_norm'] + combined['perm_norm']) / 2
            combined['importance_rank'] = combined['combined_score'].rank(ascending=False, method='min').astype(int)
            
            final_imp = combined[['feature', 'rf_mdi_importance', 'permutation_mean', 
                                  'permutation_std', 'combined_score', 'importance_rank']]
            final_imp = final_imp.sort_values('combined_score', ascending=False)
            results['Feature Importance'] = final_imp
        except Exception as e:
            logger.error(f"Feature Importance hatası: {e}")
        
        # 2. Korelasyon Anlamlılık
        try:
            logger.info("Korelasyon anlamlılık analizi hesaplanıyor...")
            pairs = self.config.correlation_pairs
            available_pairs = [(c1, c2) for c1, c2 in pairs if c1 in self.results_df.columns and c2 in self.results_df.columns]
            if available_pairs:
                sig_df = batch_correlation_analysis(
                    self.results_df, available_pairs,
                    n_bootstrap=self.config.bootstrap_n_iterations,
                    random_state=self.config.rf_random_state
                )
                results['Korelasyon Anlamlılık'] = sig_df
        except Exception as e:
            logger.error(f"Anlamlılık analizi hatası: {e}")
        
        # 3. Çok Kriterli Pareto
        try:
            logger.info("Çok Kriterli Pareto hesaplanıyor...")
            pareto_df = compute_multi_criteria_pareto(
                self.results_df,
                top_n=self.config.pareto_top_n
            )
            results['Pareto Özeti'] = pareto_df
        except Exception as e:
            logger.error(f"Pareto hatası: {e}")
        
        # 4. Executive Summary
        try:
            logger.info("Executive Summary oluşturuluyor...")
            exec_df = generate_executive_summary(
                self.results_df,
                min_trades_for_best_pf=self.config.min_trades_for_best_pf,
                min_trades_for_balanced=self.config.min_trades_for_balanced,
                max_loss_threshold=self.config.max_loss_threshold
            )
            results['Executive Summary'] = exec_df
        except Exception as e:
            logger.error(f"Executive Summary hatası: {e}")
        
        # 5. Sobol Duyarlılık Analizi
        try:
            logger.info("Sobol duyarlılık analizi hesaplanıyor...")
            from .sensitivity import compute_sobol_indices
            sobol_df = compute_sobol_indices(
                self.results_df,
                target_col=target,
                param_cols=param_cols,
                n_samples=1024,
                random_state=self.config.rf_random_state
            )
            if not sobol_df.empty:
                results['Sobol Duyarlılık'] = sobol_df
        except ImportError as e:
            logger.warning(f"Sobol analizi atlandı (SALib yüklü değil): {e}")
        except Exception as e:
            logger.error(f"Sobol analizi hatası: {e}")

        return results