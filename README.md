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
│   │   ├── patterns.py         
│   │   └── windowing.py        
│   │
│   ├── models/                 # ML Engineering Domain
│   │   ├── rnn_baseline.py     
│   │   ├── rnn_augmented.py    
│   │   └── trainer.py          # Standardized training loop for fair comparison
│   │
│   └── utils/                  # MLOps & Eval Domain
│       ├── metrics.py          
│       └── plotting.py         
│
└── reports/                    # GitHub-safe documentation
    └── figures/                # Saved charts and final visualizations
```

## Authentication for Private Repository

Because this repository is private, standard `git clone` commands in Google Colab will fail. Each team member must generate a Personal Access Token (PAT) and store it securely in Colab's Secrets manager.

### Step 1: Generate a GitHub Personal Access Token
*Note: Each team member must do this on their own GitHub account.*

1. Go to GitHub, click your profile picture in the top right, and select **Settings**.
2. Scroll down the left sidebar and click **Credentials**.
3. Click **Personal access tokens (classic)** -> **Generate new token** -> **Generate new token (classic)**.
4. Name it something recognizable (e.g., "COMP 653 Colab Token"), set an expiration date (e.g., 90 days), and check the **`repo`** box to grant access to private repositories.
5. Click generate and **copy the token**. You will not be able to see it again once you leave the page.

### Step 2: Store the Token Securely in Google Colab
*⚠️ **CRITICAL:** Never paste this token directly into notebook text or commit it to GitHub.*

1. Open your Google Colab execution notebook.
2. Click the **Key icon (🔑)** on the left sidebar to open the "Secrets" panel.
3. Click **Add new secret**.
4. Name the secret **`GITHUB_TOKEN`** (must be typed exactly like this) and paste your copied token into the Value box.
5. Toggle the **"Notebook access"** button to ON.

## Execution via Google Colab Pro
Because data is hosted on a Google Shared Drive and code is version-controlled here, all execution notebooks (`notebooks/`) must begin with the following boilerplate to bridge the environments:

```python
import os
import sys
from google.colab import drive, userdata

# 1. Mount Google Drive for Datasets & Model Weights
drive.mount('/content/drive')

# 2. Define Absolute Paths
GITHUB_TOKEN = userdata.get('GITHUB_TOKEN')
GITHUB_USER = "jesseingraham"
REPO_NAME = "comp-653-stock-prediction-i"
REPO_PATH = f"/content/{REPO_NAME}"

# 3. Clone ONLY if we haven't already
if not os.path.exists(REPO_PATH):
    !git clone https://{GITHUB_TOKEN}@github.com/{GITHUB_USER}/{REPO_NAME}.git

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
