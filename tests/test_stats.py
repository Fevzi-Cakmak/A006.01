import pytest
import pandas as pd
import numpy as np
from research.stats.significance import correlation_with_ci, batch_correlation_analysis


def test_correlation_with_ci_basic(sample_results_df):
    """Temel korelasyon ve CI hesaplanabiliyor mu?"""
    result = correlation_with_ci(sample_results_df, 'compound_return', 'profit_factor', n_bootstrap=10)
    
    assert 'error' not in result
    assert 'pearson_r' in result
    assert 'pearson_p' in result
    assert 'spearman_r' in result
    assert 'kendall_tau' in result
    assert 'ci_lower' in result
    assert 'ci_upper' in result
    assert -1 <= result['pearson_r'] <= 1
    assert 0 <= result['pearson_p'] <= 1


def test_correlation_with_ci_insufficient_data():
    """Yetersiz veri durumunda hata dönmeli."""
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    result = correlation_with_ci(df, 'a', 'b', n_bootstrap=5)
    assert 'error' in result


def test_batch_correlation_analysis(sample_results_df):
    """Toplu korelasyon analizi doğru çalışıyor mu?"""
    pairs = [
        ('compound_return', 'profit_factor'),
        ('win_rate', 'avg_return'),
    ]
    result_df = batch_correlation_analysis(sample_results_df, pairs, n_bootstrap=10)
    
    assert isinstance(result_df, pd.DataFrame)
    assert len(result_df) == len(pairs)
    assert set(result_df.columns) == {
        'pair', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p',
        'kendall_tau', 'kendall_p', 'ci_%95_lower', 'ci_%95_upper'
    }