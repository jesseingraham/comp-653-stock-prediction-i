# COMP 653 Finance Project: Two-Stage ML Model for Stock Prediction

## Overview
This project aims to build a two-stage machine learning (ML) pipeline applied to financial time series data, specifically targeting the S&P 500 index. The core problem is twofold: first, to automatically detect and classify recurring chart patterns in OHLCV (Open, High, Low, Close, Volume) price data as they emerge in real time; and second, to leverage those pattern classifications as contextual features to improve intraday price forecasting. This work is motivated by the practical goal of developing a data-driven trading system.

## Repository Structure

To prevent Jupyter Notebook merge conflicts, core logic is maintained in modular Python scripts within the `src/` directory, while notebooks are used strictly for execution and visualization.

```text
comp-653-stock-prediction-i/
│
├── README.md                   # Project overview and instructions
├── requirements.txt            # Shared libraries (yfinance, torch/tensorflow, zeta-zetra)
├── config.py                   # Global variables (Drive paths, tickers, look-back windows)
│
├── notebooks/                  # Colab Notebooks (Execution dashboards)
│   ├── 01_data_collection.ipynb       
│   ├── 02_feature_engineering.ipynb   
│   ├── 03_baseline_model.ipynb        
│   ├── 04_augmented_model.ipynb       
│   └── 05_evaluation.ipynb            
│
├── src/                        # Core Python Modules
│   ├── __init__.py
│   │
│   ├── data/                   # Data Architect Domain
│   │   ├── fetcher.py          
│   │   └── cleaner.py          
│   │
│   ├── features/               # Feature Engineering Domain
│   │   ├── base_features.py    # Stationary OHLCV-derived inputs
│   │   ├── patterns.py         
│   │   └── windowing.py        # Look-back windows + log-return targets
│   │
│   ├── models/                 # ML Engineering Domain
│   │   ├── rnn_forecaster.py   # One RNN for both arms (features are the variable)
│   │   └── trainer.py          # Standardized training loop for fair comparison
│   │
│   └── utils/                  # MLOps & Eval Domain
│       ├── artifacts.py        # Save/load complete runs to the Shared Drive
│       ├── metrics.py          
│       └── plotting.py         
│
└── reports/                    # GitHub-safe documentation
    └── figures/                # Saved charts and final visualizations
```
## Execution via Google Colab Pro
Because data is hosted on a Google Shared Drive and code is version-controlled here, all execution notebooks (`notebooks/`) must begin with the following boilerplate to bridge the environments:

```python
import os
import sys
from google.colab import drive, userdata

# 1. Mount Google Drive for Datasets & Model Weights
drive.mount('/content/drive')

# 2. Define Absolute Paths
#GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
GITHUB_USER = "jesseingraham"
REPO_NAME = "comp-653-stock-prediction-i"
REPO_PATH = f"/content/{REPO_NAME}"

# 3. Clone ONLY if we haven't already
if not os.path.exists(REPO_PATH):
    #!git clone https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git
    !git clone https://github.com/{GITHUB_USER}/{REPO_NAME}.git

# 4. Safely change to the absolute path
os.chdir(REPO_PATH)

# 5. Add the repository root to Python's system path
if REPO_PATH not in sys.path:
    sys.path.append(REPO_PATH)

# 6. Install dependencies
!pip install -r requirements.txt

# 7. Import modularized code
from config import DRIVE_DATA_PATH
# from src.data.fetcher import get_sp500_data
```
