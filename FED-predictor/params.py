# Global project params

import os

# ENCODER.PY
RENAME_RATES = {'release date': 'date', 'actual': 'rate'}
TEST_SIZE = 0.2 
RANDOM_STATE = 42

# MODEL.PY
HIDDEN_DIM = 128
OUTPUT_DIM = 3  # For multiclass classification (up, down, no change)
K = 5 
LEARNING_RATE = 0.001
EPOCHS = 10

TEXT_FILE = 'Fed_Scrape-2015-2023.csv'
RATES_FILE= 'US Fed Rate.csv'

RAW_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "aferri-git", "FED-Predictor", 'data', 'raw')
PROCESSED_DATA_PATH = os.path.join(os.path.expanduser('~'), "code", "aferri-git", "FED-Predictor", 'data', 'processed')
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'), "code", "aferri-git", "FED-Predictor", "training_outputs")