from research.constants import COLUMN_NAME_MAP

def test_column_name_map_has_all_keys():
    """Tüm beklenen sütunlar mapping'te var mı?"""
    expected_keys = {
        'compound_return', 'win_rate', 'total_trades', 'profit_factor',
        'max_win', 'max_loss', 'avg_return', 'stop_loss_ratio',
        'trailing_stop_ratio', 'rsi_low', 'adx_strong', 'volume_high_ratio'
    }
    assert set(COLUMN_NAME_MAP.keys()) == expected_keys

def test_column_name_map_values_are_strings():
    """Tüm mapping değerleri string olmalı."""
    assert all(isinstance(v, str) for v in COLUMN_NAME_MAP.values())
    # Türkçe karakter içermeli (kullanıcıya gösterilecek)
    assert any('ı' in v or 'ğ' in v or 'ü' in v for v in COLUMN_NAME_MAP.values())