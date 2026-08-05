# research/sampling/test_sampling.py
import pytest
from .sampling import RandomSampler, LatinHypercubeSampler, SobolSampler, GridSampler, SamplerFactory

def test_random_sampler():
    param_space = {
        'stop_loss': {'low': 0.08, 'high': 0.12, 'dist': 'uniform'},
        'adx': {'values': [28, 30, 32, 35], 'dist': 'choice'},
    }
    sampler = RandomSampler(param_space, seed=42)
    samples = sampler.sample(10)
    assert len(samples) == 10
    for s in samples:
        assert 'stop_loss' in s
        assert 'adx' in s
        assert 0.08 <= s['stop_loss'] <= 0.12
        assert s['adx'] in [28, 30, 32, 35]

def test_latin_hypercube():
    param_space = {
        'x': {'low': 0, 'high': 10, 'dist': 'uniform'},
        'y': {'low': 0, 'high': 10, 'dist': 'uniform'},
    }
    sampler = LatinHypercubeSampler(param_space, seed=42)
    samples = sampler.sample(10)
    assert len(samples) == 10
    # LHS ile örnekler homojen dağılmalı

def test_sobol_sampler():
    param_space = {
        'x': {'low': 0, 'high': 1, 'dist': 'uniform'},
        'y': {'low': 0, 'high': 1, 'dist': 'uniform'},
    }
    try:
        sampler = SobolSampler(param_space, seed=42)
        samples = sampler.sample(10)
        assert len(samples) == 10
    except ImportError:
        pytest.skip("scipy kurulu değil")

def test_grid_sampler():
    param_space = {
        'a': {'low': 0, 'high': 1, 'dist': 'uniform'},
        'b': {'values': [1, 2, 3], 'dist': 'choice'},
    }
    sampler = GridSampler(param_space, seed=42)
    samples = sampler.sample()
    # 5 * 3 = 15 kombinasyon
    assert len(samples) == 15

def test_factory():
    param_space = {'x': {'low': 0, 'high': 1, 'dist': 'uniform'}}
    sampler = SamplerFactory.create("random", param_space)
    assert isinstance(sampler, RandomSampler)
    sampler = SamplerFactory.create("latin_hypercube", param_space)
    assert isinstance(sampler, LatinHypercubeSampler)
    with pytest.raises(ValueError):
        SamplerFactory.create("unknown", param_space)