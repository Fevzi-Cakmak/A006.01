# research/sensitivity/test_sensitivity.py
import pytest
from .sensitivity import OATAnalyzer, MorrisAnalyzer, SobolAnalyzer, SensitivityFactory


def test_oat_analyzer():
    param_space = {
        'x': {'low': 0, 'high': 10, 'dist': 'uniform'},
        'y': {'low': 0, 'high': 10, 'dist': 'uniform'},
    }
    analyzer = OATAnalyzer(param_space, seed=42)

    def objective(params):
        return params['x'] + params['y']

    base = {'x': 5, 'y': 5}
    result = analyzer.analyze(objective, base_params=base, n_points=5)

    assert result.method == "OAT"
    assert 'x' in result.values
    assert 'y' in result.values
    assert len(result.values['x']['points']) >= 3


def test_morris_analyzer():
    param_space = {
        'x': {'low': 0, 'high': 10, 'dist': 'uniform'},
        'y': {'low': 0, 'high': 10, 'dist': 'uniform'},
    }
    analyzer = MorrisAnalyzer(param_space, seed=42)

    def objective(params):
        return params['x'] + params['y']

    result = analyzer.analyze(objective, n_trajectories=5)
    assert result.method == "Morris"
    assert 'mu_star' in result.values


def test_sobol_analyzer():
    param_space = {
        'x': {'low': 0, 'high': 10, 'dist': 'uniform'},
        'y': {'low': 0, 'high': 10, 'dist': 'uniform'},
    }
    try:
        analyzer = SobolAnalyzer(param_space, seed=42)

        def objective(params):
            return params['x'] + params['y']

        result = analyzer.analyze(objective, n_samples=100)
        assert result.method == "Sobol"
        assert 'S1' in result.values
    except ImportError:
        pytest.skip("scipy kurulu değil")


def test_factory():
    param_space = {'x': {'low': 0, 'high': 1}}
    oat = SensitivityFactory.create("oat", param_space)
    assert isinstance(oat, OATAnalyzer)

    with pytest.raises(ValueError):
        SensitivityFactory.create("unknown", param_space)