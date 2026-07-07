import os

# ==========================================
# FILE PATHS (DEFAULT)
# ==========================================
# Direct path to the Google Shared Drive
DRIVE_ROOT = "/content/drive/Shareddrives/COMP 653 - Final Project"

# Sub-directories for organization
DRIVE_DATA_PATH = os.path.join(DRIVE_ROOT, "raw_data")
DRIVE_MODELS_PATH = os.path.join(DRIVE_ROOT, "saved_models")

# ==========================================
# GLOBAL PROJECT SETTINGS
# ==========================================
TICKER = "^GSPC"              # S&P 500 symbol in yfinance
START_DATE = "2006-01-01"     # ~20 years of data
END_DATE = "2026-01-01"

# Safety check for debugging
if not os.path.exists(DRIVE_ROOT):
    print(f"⚠️ WARNING: Could not find {DRIVE_ROOT}. Did you run drive.mount('/content/drive')?")
