"""Global configuration for the COMP 653 stock-prediction project.

Defines the Shared Drive paths, target index, and date range shared across the
`src/` modules and execution notebooks. Import these values from here rather
than hardcoding paths or tickers elsewhere.
"""

import os

# ==========================================
# FILE PATHS (DEFAULT)
# ==========================================
# Direct path to the Google Shared Drive.
DRIVE_ROOT = "/content/drive/Shareddrives/COMP 653 - Final Project"

# Sub-directories for organization. Raw and cleaned datasets share this data
# folder and are distinguished by filename suffix (_raw / _clean).
DRIVE_DATA_PATH = os.path.join(DRIVE_ROOT, "data")
DRIVE_MODELS_PATH = os.path.join(DRIVE_ROOT, "saved_models")

# ==========================================
# GLOBAL PROJECT SETTINGS
# ==========================================
TICKER = "^GSPC"  # S&P 500 index symbol in yfinance.
START_DATE = "2006-01-01"  # ~20 years of history.
END_DATE = "2026-01-01"

# Safety check for debugging: warn (without crashing) when the Shared Drive is
# not mounted. Plain ASCII keeps this printable on non-UTF-8 consoles.
if not os.path.exists(DRIVE_ROOT):
    print(
        f"WARNING: Could not find {DRIVE_ROOT}. "
        "Did you run drive.mount('/content/drive')?"
    )
