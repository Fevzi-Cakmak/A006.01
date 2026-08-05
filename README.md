# VIOS – Backtest & Quant Research Framework (A006)

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Modüler backtest motoru ve quant research platformu.

---

# 🎯 Amaç

VIOS;

- Backtest
- Portföy Simülasyonu
- Quant Research

için geliştirilmiş modüler Python framework'üdür.

---

# 📦 Kurulum

```bash
git clone https://github.com/<YOUR_USERNAME>/VIOS.git

cd VIOS

python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

---

# ▶️ Hızlı Başlangıç

### Excel

```bash
python run_excel.py
```

### Borsapy

```bash
python run_borsapy.py
```

### Quant Research

```bash
python run_research.py
```

---

# 🗂️ Proje Yapısı

```text
VIOS/
│
├── engine/
│
├── research/
│   ├── config/
│   ├── sampling/
│   ├── sensitivity/
│   ├── importance/
│   ├── stats/
│   ├── plotting/
│   └── reporting/
│
├── run_excel.py
├── run_borsapy.py
├── run_research.py
└── README.md
```

---

# ⚙️ Engine Pipeline

```text
DataFetcher
      │
      ▼
Indicators
      │
      ▼
Signals
      │
      ▼
Portfolio
      │
      ▼
Excel Reports
```

---

# 🔬 Research Pipeline

```text
Sampling
      │
      ▼
run_backtest()
      │
      ▼
Statistics
      │
      ▼
Correlation
      │
      ▼
Importance
      │
      ▼
Excel / HTML / JSON
```

---

# 🧩 Özellikler

## Engine

- Excel
- Borsapy
- RSI
- ADX
- MACD
- ATR
- Bollinger
- Portfolio Simulation
- Excel Reports

## Research

- Random Sampling
- Latin Hypercube
- Grid Search
- OAT
- Morris (hazır)
- Sobol (hazır)
- Pearson
- Spearman
- Kendall
- Random Forest
- Permutation Importance
- Bootstrap
- Walk Forward
- Sharpe
- Sortino
- Calmar
- VaR
- CVaR
- Heatmap
- Scatter
- Pairplot
- Tornado
- Violin
- HTML Report
- JSON Export

---

# 📊 Çıktılar

Engine

```
VIOS_A003_EXCEL_*.xlsx
```

Research

```
research_summary_*.xlsx
best_params_*.json
research_report_*.html
```

---

# 🧪 Geliştirme

```bash
black .
mypy .
pytest
pytest --cov=. --cov-report=html
```

---

# 🗺️ Roadmap

- [x] Random Sampling
- [x] Latin Hypercube
- [x] Grid
- [ ] Sobol
- [ ] Morris
- [x] Permutation Importance
- [ ] HTML Dashboard
- [ ] GitHub Actions
- [ ] pyproject.toml

---

# 🏗️ Tasarım

- SOLID
- Adapter Pattern
- Single Source of Truth
- Configuration Driven Architecture
- Engine bağımsız
- Research bağımlı

---

# 📜 Sürüm

| Sürüm | Açıklama |
|--------|----------|
| A003 | Modüler Backtest |
| A004 | Walk Forward + Bootstrap |
| A005 | Research |
| A006 | Quant Research Framework |

---

# 📄 Lisans

Bu proje özel lisans altındadır.