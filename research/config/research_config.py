from dataclasses import dataclass, field
from typing import Dict, List, Tuple

@dataclass
class ResearchConfig:
    # Mevcut alanlar...
    param_space: Dict[str, tuple] = field(default_factory=dict)
    num_samples: int = 100
    sampling_method: str = "random"
    random_seed: int = 42
    output_dir: str = "research_outputs"
    # ASCII uyumlu hedef değişken
    target_metric: str = "compound_return"
    parallel_workers: int = 4
    save_individual_reports: bool = False
    
    # Yeni Parametreler (P1, P2, P3)
    rf_n_estimators: int = 500
    rf_random_state: int = 42
    permutation_n_repeats: int = 10
    bootstrap_n_iterations: int = 1000
    pareto_top_n: int = 20
    
    # Executive Summary eşikleri
    min_trades_for_best_pf: int = 30
    min_trades_for_balanced: int = 40
    max_loss_threshold: float = -14.0
    
    # Korelasyon çiftleri
    correlation_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('avg_return', 'profit_factor'),
        ('win_rate', 'avg_return'),
        ('win_rate', 'profit_factor'),
        ('stop_loss_ratio', 'max_loss'),
        ('rsi_low', 'total_trades'),
        ('volume_high_ratio', 'compound_return'),
    ])
    
    log_level: str = "INFO"