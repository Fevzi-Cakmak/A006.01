# research/plotting/test_plotting.py
import pytest
import pandas as pd
import numpy as np
import os
from .plotting import (
    ScatterPlotter,
    HeatmapPlotter,
    TornadoPlotter,
    DistributionPlotter,
    PairPlotter,
    PlotterFactory,
    create_scatter,
    create_heatmap,
    create_tornado,
    create_distribution,
)


def test_scatter_plotter():
    data = pd.DataFrame({
        'param1': np.random.randn(100),
        'param2': np.random.randn(100),
        'getiri': np.random.randn(100) * 10,
    })
    plotter = ScatterPlotter(output_dir="test_plots")
    filepath = plotter.plot(data, x_col='param1', y_col='getiri')
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_heatmap_plotter():
    try:
        import seaborn
    except ImportError:
        pytest.skip("seaborn kurulu değil")

    data = pd.DataFrame({
        'a': np.random.randn(50),
        'b': np.random.randn(50),
        'c': np.random.randn(50),
    })
    plotter = HeatmapPlotter(output_dir="test_plots")
    filepath = plotter.plot(data, columns=['a', 'b', 'c'])
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_tornado_plotter():
    values = {'a': 0.8, 'b': 0.5, 'c': 0.3, 'd': 0.1}
    plotter = TornadoPlotter(output_dir="test_plots")
    filepath = plotter.plot(values)
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_distribution_plotter():
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100) * 2,
    })
    plotter = DistributionPlotter(output_dir="test_plots")
    filepath = plotter.plot(data, kind='kde')
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_pairplot_plotter():
    try:
        import seaborn
    except ImportError:
        pytest.skip("seaborn kurulu değil")

    data = pd.DataFrame({
        'a': np.random.randn(50),
        'b': np.random.randn(50),
        'c': np.random.randn(50),
        'getiri': np.random.randn(50) * 5,
    })
    plotter = PairPlotter(output_dir="test_plots")
    filepath = plotter.plot(data, columns=['a', 'b', 'c'])
    assert os.path.exists(filepath)
    os.remove(filepath)


def test_factory():
    plotter = PlotterFactory.create("scatter", output_dir="test_plots")
    assert isinstance(plotter, ScatterPlotter)

    plotter = PlotterFactory.create("heatmap", output_dir="test_plots")
    assert isinstance(plotter, HeatmapPlotter)

    with pytest.raises(ValueError):
        PlotterFactory.create("unknown")