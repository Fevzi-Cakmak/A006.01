# research/importance/test_importance.py
import pytest
import pandas as pd
import numpy as np
from .importance import (
    PearsonImportance,
    SpearmanImportance,
    KendallImportance,
    RandomForestImportance,
    PermutationImportance,
    ImportanceFactory,
)

@pytest.fixture
def sample_df():
    """Örnek veri seti."""
    np.random.seed(42)
    n = 50
    data = {
        'param1': np.random.uniform(0, 1, n),
        'param2': np.random.uniform(0, 1, n),
        'param3': np.random.choice([1, 2, 3], n),
        'getiri': np.random.normal(10, 5, n),
    }
    # Bilinen ilişki ekleyelim
    data['getiri'] = data['getiri'] + 2 * data['param1'] + 0.5 * data['param2']
    return pd.DataFrame(data)

def test_pearson(sample_df):
    analyzer = PearsonImportance()
    result = analyzer.analyze(sample_df)
    assert len(result) == 3
    assert 'param1' in result.index
    assert result['param1'] > result['param2']  # param1 daha etkili

def test_spearman(sample_df):
    analyzer = SpearmanImportance()
    result = analyzer.analyze(sample_df)
    assert len(result) == 3

def test_kendall(sample_df):
    analyzer = KendallImportance()
    result = analyzer.analyze(sample_df)
    assert len(result) == 3

def test_random_forest(sample_df):
    analyzer = RandomForestImportance(n_estimators=10, random_state=42)
    result = analyzer.analyze(sample_df)
    assert len(result) == 3
    # Önem değerleri toplamı 1 olmalı
    assert abs(result.sum() - 1.0) < 0.01

def test_permutation(sample_df):
    analyzer = PermutationImportance(n_repeats=3, random_state=42, n_estimators=10)
    result = analyzer.analyze(sample_df)
    assert len(result) == 3

def test_factory():
    analyzer = ImportanceFactory.create("pearson")
    assert isinstance(analyzer, PearsonImportance)
    analyzer = ImportanceFactory.create("random_forest", n_estimators=50)
    assert isinstance(analyzer, RandomForestImportance)
    with pytest.raises(ValueError):
        ImportanceFactory.create("unknown")