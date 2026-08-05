import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def add_heatmap(results_df, tmpdir):
    """
    Korelasyon heatmap oluşturur.
    
    Args:
        results_df: Araştırma sonuçlarını içeren DataFrame
        tmpdir: Geçici dosya dizini
    
    Returns:
        list: [(sayfa_adı, dosya_yolu), ...] formatında liste
    """
    if len(results_df) < 2:
        return []
    
    try:
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return []
        
        corr = results_df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            ax=ax,
            square=True,
            cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Korelasyon Heatmap", fontsize=14)
        plt.tight_layout()
        
        filepath = os.path.join(tmpdir, 'heatmap.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(filepath):
            return [('Korelasyon Heatmap', filepath)]
        
    except Exception as e:
        logger.warning(f"Heatmap oluşturulamadı: {e}")
    
    return []