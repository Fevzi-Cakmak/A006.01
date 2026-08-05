import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def add_pairplot(results_df, config, tmpdir):
    """
    Parametre dağılımları pairplot oluşturur.
    
    Args:
        results_df: Araştırma sonuçlarını içeren DataFrame
        config: ResearchConfig nesnesi
        tmpdir: Geçici dosya dizini
    
    Returns:
        list: [(sayfa_adı, dosya_yolu), ...] formatında liste
    """
    # Çok fazla veri varsa performans sorunu yaşanabilir
    if len(results_df) > 100:
        logger.info("Pairplot atlandı (veri sayısı 100'den fazla).")
        return []
    
    try:
        param_cols = list(config.param_space.keys())
        if len(param_cols) < 2:
            return []
        
        # ASCII uyumlu sütun adı
        plot_data = results_df[param_cols + ['compound_return']].copy()
        
        # Outlier temizliği (grafik okunabilirliği için)
        for col in plot_data.columns:
            q1 = plot_data[col].quantile(0.01)
            q3 = plot_data[col].quantile(0.99)
            plot_data[col] = plot_data[col].clip(q1, q3)
        
        g = sns.pairplot(
            plot_data,
            diag_kind='kde',
            plot_kws={'alpha': 0.5, 's': 20},
            diag_kws={'fill': True}
        )
        g.fig.suptitle("Parametre Dağılımları", y=1.02, fontsize=14)
        
        filepath = os.path.join(tmpdir, 'pairplot.png')
        g.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(g.fig)
        
        if os.path.exists(filepath):
            return [('Parametre Dağılımları', filepath)]
        
    except Exception as e:
        logger.warning(f"Pairplot oluşturulamadı: {e}")
    
    return []