# research/plotting/plotting.py
"""
Görselleştirme araçları.

Matplotlib ve Seaborn kullanır (opsiyonel).
"""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # GUI olmadan çalıştırmak için
except ImportError:
    plt = None
    warnings.warn("matplotlib kurulu değil. Görseller oluşturulamaz.")

try:
    import seaborn as sns
except ImportError:
    sns = None
    warnings.warn("seaborn kurulu değil. Bazı görseller (violin, pairplot) kullanılamaz.")


class BasePlotter(ABC):
    """Tüm görselleştiriciler için temel sınıf."""

    def __init__(self, output_dir: str = "plots", dpi: int = 150, figsize: Tuple[int, int] = (10, 6)):
        self.output_dir = output_dir
        self.dpi = dpi
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def plot(self, data: pd.DataFrame, **kwargs) -> str:
        """Görsel oluşturur ve dosya yolunu döndürür."""
        pass

    def _save(self, fig, filename: str) -> str:
        """Figürü kaydeder ve dosya yolunu döndürür."""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return filepath


class ScatterPlotter(BasePlotter):
    """Scatter plot: Parametre vs getiri."""

    def plot(
        self,
        data: pd.DataFrame,
        x_col: str,
        y_col: str = "getiri",
        color_col: Optional[str] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        alpha: float = 0.6,
        **kwargs,
    ) -> str:
        fig, ax = plt.subplots(figsize=self.figsize)

        if color_col and color_col in data.columns:
            scatter = ax.scatter(
                data[x_col],
                data[y_col],
                c=data[color_col],
                alpha=alpha,
                cmap="viridis",
                s=30,
            )
            plt.colorbar(scatter, ax=ax, label=color_col)
        else:
            ax.scatter(data[x_col], data[y_col], alpha=alpha, s=30, color="steelblue")

        ax.set_xlabel(xlabel or x_col)
        ax.set_ylabel(ylabel or y_col)
        ax.set_title(title or f"{x_col} vs {y_col}")
        ax.grid(True, alpha=0.3)

        filename = f"scatter_{x_col}.png"
        return self._save(fig, filename)


class HeatmapPlotter(BasePlotter):
    """Korelasyon heatmap'i."""

    def plot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        title: str = "Korelasyon Matrisi",
        cmap: str = "coolwarm",
        annot: bool = True,
        fmt: str = ".2f",
        **kwargs,
    ) -> str:
        if sns is None:
            raise ImportError("seaborn kurulu değil. Heatmap için seaborn gerekli.")

        if columns is None:
            columns = data.columns.tolist()

        corr = data[columns].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8},
            linewidths=0.5,
        )
        ax.set_title(title)

        return self._save(fig, "heatmap.png")


class TornadoPlotter(BasePlotter):
    """Tornado chart - duyarlılık sıralaması."""

    def plot(
        self,
        values: Union[pd.Series, Dict[str, float]],
        title: str = "Parametre Duyarlılığı",
        xlabel: str = "Etki",
        sort: bool = True,
        color: str = "steelblue",
        **kwargs,
    ) -> str:
        if isinstance(values, dict):
            values = pd.Series(values)

        if sort:
            values = values.sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=self.figsize)
        bars = ax.barh(values.index, values.values, color=color)

        # Değerleri etiket olarak ekle
        for bar, val in zip(bars, values.values):
            ax.text(val + 0.02 * max(values.values), bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=9)

        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="x")

        return self._save(fig, "tornado.png")


class DistributionPlotter(BasePlotter):
    """Parametre dağılımı (KDE, Box, Violin)."""

    def plot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        kind: str = "kde",  # kde, box, violin, histogram
        title: Optional[str] = None,
        **kwargs,
    ) -> str:
        if columns is None:
            columns = [c for c in data.columns if data[c].dtype.kind in "if"]

        if not columns:
            raise ValueError("Sayısal sütun bulunamadı.")

        n_cols = min(3, len(columns))
        n_rows = (len(columns) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for i, col in enumerate(columns):
            ax = axes[i] if i < len(axes) else None
            if ax is None:
                break

            if kind == "kde":
                if sns is not None:
                    sns.kdeplot(data=data, x=col, ax=ax, fill=True, color="steelblue")
                else:
                    ax.hist(data[col].dropna(), bins=30, density=True, alpha=0.7)
                    ax.set_ylabel("Yoğunluk")
            elif kind == "box":
                if sns is not None:
                    sns.boxplot(data=data, y=col, ax=ax, color="steelblue")
                else:
                    ax.boxplot(data[col].dropna(), vert=True)
            elif kind == "violin":
                if sns is not None:
                    sns.violinplot(data=data, y=col, ax=ax, color="steelblue")
                else:
                    ax.boxplot(data[col].dropna(), vert=True)
                    ax.set_title("Violin (boxplot yedek)")
            else:  # histogram
                ax.hist(data[col].dropna(), bins=30, alpha=0.7, color="steelblue", edgecolor="white")

            ax.set_title(col)
            ax.grid(True, alpha=0.3)

        # Kullanılmayan eksenleri gizle
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        return self._save(fig, f"distribution_{kind}.png")


class PairPlotter(BasePlotter):
    """Pair plot - çoklu parametre ilişkileri."""

    def plot(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        hue: Optional[str] = None,
        title: Optional[str] = None,
        **kwargs,
    ) -> str:
        if sns is None:
            raise ImportError("seaborn kurulu değil. Pair plot için seaborn gerekli.")

        if columns is None:
            # Sayısal sütunları al, getiri de dahil
            numeric_cols = [c for c in data.columns if data[c].dtype.kind in "if"]
            if "getiri" in numeric_cols:
                # getiri'yi sona koy
                numeric_cols.remove("getiri")
                columns = numeric_cols + ["getiri"]
            else:
                columns = numeric_cols[:6]  # En fazla 6 sütun

        if len(columns) < 2:
            raise ValueError("Pair plot için en az 2 sütun gerekli.")

        # Diag KDE veya histogram
        g = sns.pairplot(data[columns], diag_kind="kde", plot_kws={"alpha": 0.5, "s": 10})
        if title:
            g.fig.suptitle(title, y=1.02)

        return self._save(g.fig, "pairplot.png")


class PlotterFactory:
    """Görselleştirici fabrikası."""

    @staticmethod
    def create(plot_type: str, output_dir: str = "plots", **kwargs) -> BasePlotter:
        if plot_type == "scatter":
            return ScatterPlotter(output_dir, **kwargs)
        elif plot_type == "heatmap":
            return HeatmapPlotter(output_dir, **kwargs)
        elif plot_type == "tornado":
            return TornadoPlotter(output_dir, **kwargs)
        elif plot_type == "distribution":
            return DistributionPlotter(output_dir, **kwargs)
        elif plot_type == "pairplot":
            return PairPlotter(output_dir, **kwargs)
        else:
            raise ValueError(f"Bilinmeyen plot tipi: {plot_type}")


# ---------- Kısayol fonksiyonları ----------

def create_scatter(data, x_col, y_col="getiri", output_dir="plots", **kwargs):
    plotter = ScatterPlotter(output_dir)
    return plotter.plot(data, x_col, y_col, **kwargs)


def create_heatmap(data, columns=None, output_dir="plots", **kwargs):
    plotter = HeatmapPlotter(output_dir)
    return plotter.plot(data, columns, **kwargs)


def create_tornado(values, output_dir="plots", **kwargs):
    plotter = TornadoPlotter(output_dir)
    return plotter.plot(values, **kwargs)


def create_distribution(data, columns=None, output_dir="plots", **kwargs):
    plotter = DistributionPlotter(output_dir)
    return plotter.plot(data, columns, **kwargs)


def create_pairplot(data, columns=None, output_dir="plots", **kwargs):
    plotter = PairPlotter(output_dir)
    return plotter.plot(data, columns, **kwargs)