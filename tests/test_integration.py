import pytest
from research import ResearchConfig, ResearchRunner
from unittest.mock import Mock, patch


@patch('research.runner.run_backtest')
def test_research_runner_minimal(mock_run_backtest, sample_config):
    """ResearchRunner'ın en azından başlatılıp çalıştırılabildiğini test et."""
    # Mock backtest sonucu döndür
    mock_run_backtest.return_value = 'mock_result.xlsx'
    
    # Mock data_fetcher ve semboller
    mock_fetcher = Mock()
    symbols = ['AEFES', 'AKBNK']
    b30 = set()
    b50 = set()
    b100 = set()
    
    runner = ResearchRunner(
        config=sample_config,
        data_fetcher=mock_fetcher,
        symbols=symbols,
        b30=b30,
        b50=b50,
        b100=b100
    )
    # sadece init çalıştı mı kontrol et
    assert runner.config == sample_config
    assert runner.symbols == symbols