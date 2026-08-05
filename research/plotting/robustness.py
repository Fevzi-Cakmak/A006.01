import os
import logging
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def add_robustness(results_df, tmpdir):
    """
    CV (Coefficient of Variation) bazlı robustness bar grafiği oluşturur.
    
    Args:
        results_df: Araştırma sonuçlarını içeren DataFrame
        tmpdir: Geçici dosya dizini
    
    Returns:
        list: [(sayfa_adı, dosya_yolu), ...] formatında liste
    """
    try:
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return []
        
        summary = results_df[numeric_cols].describe().T
        if 'cv' not in summary.columns:
            summary['cv'] = summary['std'] / summary['mean']
        
        cv_data = summary['cv'].dropna()
        if cv_data.empty:
            return []
        
        fig, ax = plt.subplots(figsize=(10, 6))
        cv_data.plot(kind='bar', ax=ax, color='steelblue')
        ax.set_title('Parametre Robustness (CV)', fontsize=14)
        ax.set_ylabel('Coefficient of Variation', fontsize=12)
        ax.set_xlabel('Parametreler', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        filepath = os.path.join(tmpdir, 'robustness.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        if os.path.exists(filepath):
            return [('Robustness (CV)', filepath)]
        
    except Exception as e:
        logger.warning(f"Robustness grafiği oluşturulamadı: {e}")
    
    return []