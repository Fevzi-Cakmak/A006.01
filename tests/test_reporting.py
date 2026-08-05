import pandas as pd
import numpy as np
import pytest
from research.reporting.pareto import compute_multi_criteria_pareto
from research.reporting.executive_summary import generate_executive_summary


def test_pareto_output():
    """Pareto fonksiyonunun doğru çıktı verdiğini kontrol eder."""
    np.random.seed(42)
    df = pd.DataFrame({
        'compound_return': np.random.uniform(100, 1000, 50),
        'avg_return': np.random.uniform(2, 10, 50),
        'profit_factor': np.random.uniform(1.5, 4.0, 50),
        'win_rate': np.random.uniform(40, 80, 50),
        'max_loss': np.random.uniform(-20, -5, 50),
        'stop_loss_ratio': np.random.uniform(0.08, 0.12, 50),
        'trailing_stop_ratio': np.random.uniform(0.05, 0.10, 50),
        'rsi_low': np.random.uniform(55, 65, 50),
        'adx_strong': np.random.uniform(28, 35, 50),
        'volume_high_ratio': np.random.uniform(1.8, 2.5, 50),
    })
    
    result = compute_multi_criteria_pareto(df, top_n=20)
    
    expected_columns = ['Metrik', 'Top20_Ortalama', 'TumVeri_Ortalama', 'Fark']
    assert all(col in result.columns for col in expected_columns)
    
    expected_metrics = ['compound_return', 'avg_return', 'profit_factor', 'win_rate', 'max_loss']
    assert set(result['Metrik']) == set(expected_metrics)
    
    assert pd.api.types.is_numeric_dtype(result['Top20_Ortalama'])
    assert pd.api.types.is_numeric_dtype(result['TumVeri_Ortalama'])
    assert pd.api.types.is_numeric_dtype(result['Fark'])
    
    for metric in ['compound_return', 'avg_return', 'profit_factor', 'win_rate']:
        top20_mean = result.loc[result['Metrik'] == metric, 'Top20_Ortalama'].values[0]
        assert top20_mean >= 0, f"{metric} için Top20_Ortalama negatif: {top20_mean}"
    
    max_loss_mean = result.loc[result['Metrik'] == 'max_loss', 'Top20_Ortalama'].values[0]
    assert not np.isnan(max_loss_mean), "max_loss Top20_Ortalama NaN"
    
    for metric in expected_metrics:
        top20 = result.loc[result['Metrik'] == metric, 'Top20_Ortalama'].values[0]
        tum = result.loc[result['Metrik'] == metric, 'TumVeri_Ortalama'].values[0]
        fark = result.loc[result['Metrik'] == metric, 'Fark'].values[0]
        assert abs(fark - (top20 - tum)) < 1e-6, f"{metric} için Fark yanlış hesaplanmış"


def test_executive_summary_empty():
    """Boş DataFrame ile Executive Summary çağrıldığında boş DataFrame dönmeli."""
    df_empty = pd.DataFrame()
    result = generate_executive_summary(df_empty)
    assert result.empty, "Boş DataFrame beklenirken dolu DataFrame döndü"


def test_executive_summary_missing_column():
    """Eksik sütun durumunda ValueError fırlatılmalı."""
    df = pd.DataFrame({
        'compound_return': [100, 200],
        'profit_factor': [2.0, 3.0],
        # 'max_loss' sütunu eksik
    })
    with pytest.raises(ValueError, match="Eksik sütunlar"):
        generate_executive_summary(df)