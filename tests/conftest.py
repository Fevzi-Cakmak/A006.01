import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def sample_results_df():
    """5 satırlık örnek backtest sonuçları."""
    np.random.seed(42)
    data = {
        'stop_loss_ratio': np.random.uniform(0.08, 0.12, 5),
        'trailing_stop_ratio': np.random.uniform(0.05, 0.10, 5),
        'rsi_low': np.random.randint(55, 65, 5),
        'adx_strong': np.random.randint(28, 35, 5),
        'volume_high_ratio': np.random.uniform(1.8, 2.5, 5),
        'compound_return': np.random.uniform(100, 500, 5),
        'win_rate': np.random.uniform(50, 70, 5),
        'profit_factor': np.random.uniform(1.5, 3.0, 5),
        'total_trades': np.random.randint(10, 50, 5),
        'max_win': np.random.uniform(10, 25, 5),
        'max_loss': np.random.uniform(-20, -5, 5),
        'avg_return': np.random.uniform(2, 8, 5),
    }
    df = pd.DataFrame(data)
    # Gerçekçi olması için max_loss negatif olsun
    df['max_loss'] = -np.abs(df['max_loss'])
    return df

@pytest.fixture
def sample_config():
    from research.config.research_config import ResearchConfig
    config = ResearchConfig(
        param_space={
            'stop_loss_ratio': {'low': 0.08, 'high': 0.12},
            'trailing_stop_ratio': {'low': 0.05, 'high': 0.10},
            'rsi_low': {'low': 55, 'high': 65},
            'adx_strong': {'low': 28, 'high': 35},
            'volume_high_ratio': {'low': 1.8, 'high': 2.5},
        },
        num_samples=10,
        random_seed=42,
    )
    return config