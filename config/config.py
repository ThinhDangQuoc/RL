import pathlib

#import finrl

import pandas as pd
import datetime
import os
#pd.options.display.max_rows = 10
#pd.options.display.max_columns = 10


#PACKAGE_ROOT = pathlib.Path(finrl.__file__).resolve().parent
#PACKAGE_ROOT = pathlib.Path().resolve().parent

#TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
#DATASET_DIR = PACKAGE_ROOT / "data"

# data
#TRAINING_DATA_FILE = "data/ETF_SPY_2009_2020.csv"
TRAINING_DATA_FILE = "data/processed2.csv"
# TRAINING_DATA_FILE = "data/dow_30_2009_2020.csv"

# Ticker universe
# Your dataset does not have full 30-ticker coverage in 2014-2020.
# To avoid heavy synthetic filling and distribution shift, train on tickers
# that are sufficiently present in that period.
# These tickers have >= 95% date coverage in 2014-2020.
TICKERS = [
	"STB",
	"BVH",
	"FPT",
	"MBB",
	"MSN",
	"CTG",
	"SHB",
	"SSI",
	"GAS",
	"HPG",
	"VCB",
	"VIC",
	"VNM",
	"ACB",
	"BID",
]

STOCK_DIM = len(TICKERS)

# Date ranges (YYYYMMDD as int). Adjust these to match your dataset.
# For processed2.csv: 2014-01-02 to 2025-05-30.
DATA_START_DATE = 20140102
TRAIN_START_DATE = 20140102
TRADE_START_DATE = 20210104
TRADE_END_DATE = 20250530

now = datetime.datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
os.makedirs(TRAINED_MODEL_DIR)
TURBULENCE_DATA = "data/dow30_turbulence_index.csv"

TESTING_DATA_FILE = "data/test.csv"

# Vietnam Market Constraints
SETTLEMENT_DELAY_DAYS = 3  # T+3 settlement delay
PRICE_LIMIT_PERCENT = 0.07  # Daily price limit limit (7%)



