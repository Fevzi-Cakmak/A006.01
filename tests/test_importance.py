import pytest
import pandas as pd
import numpy as np
from research.importance.random_forest import compute_rf_importance
from research.importance.permutation import compute_permutation_importance


def test_rf_importance_output(sample_results_df):
    """RF importance doğru formatta dönüyor mu?"""
    feature_cols = ['stop_loss_ratio', 'trailing_stop_ratio', 'rsi_low', 'adx_strong', 'volume_high_ratio']
    result = compute_rf_importance(sample_results_df, 'compound_return', feature_cols)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'feature', 'rf_mdi_importance'}
    assert len(result) == len(feature_cols)
    assert result['rf_mdi_importance'].sum() == pytest.approx(1.0, rel=1e-2)


def test_permutation_importance_output(sample_results_df):
    """Permutation importance doğru formatta dönüyor mu?"""
    feature_cols = ['stop_loss_ratio', 'trailing_stop_ratio', 'rsi_low', 'adx_strong', 'volume_high_ratio']
    result = compute_permutation_importance(sample_results_df, 'compound_return', feature_cols, n_repeats=2)
    
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'feature', 'permutation_mean', 'permutation_std'}
    assert len(result) == len(feature_cols)
    assert all(result['permutation_mean'] >= 0)  # önem skorları negatif olmamalı


def test_rf_importance_stable(sample_results_df):
    """Aynı random seed ile RF importance tekrarlanabilir olmalı."""
    feature_cols = ['stop_loss_ratio', 'trailing_stop_ratio', 'rsi_low', 'adx_strong', 'volume_high_ratio']
    result1 = compute_rf_importance(sample_results_df, 'compound_return', feature_cols, random_state=42)
    result2 = compute_rf_importance(sample_results_df, 'compound_return', feature_cols, random_state=42)
    pd.testing.assert_frame_equal(result1, result2)


def test_permutation_importance_reproducible(sample_results_df):
    """Permutation importance tekrarlanabilir olmalı (random_state ile)."""
    feature_cols = ['stop_loss_ratio', 'trailing_stop_ratio', 'rsi_low', 'adx_strong', 'volume_high_ratio']
    result1 = compute_permutation_importance(sample_results_df, 'compound_return', feature_cols, 
                                             n_repeats=3, random_state=42)
    result2 = compute_permutation_importance(sample_results_df, 'compound_return', feature_cols, 
                                             n_repeats=3, random_state=42)
    pd.testing.assert_frame_equal(result1, result2)