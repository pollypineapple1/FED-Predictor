# Global project params

import os

# ENCODER.PY
RENAME_RATES = {'Release Date': 'date', 'Actual': 'rate'}
TEST_SIZE=0.2 
RANDOM_STATE=42

# MODEL.PY
HIDDEN_DIM = 128
OUTPUT_DIM = 3  # For multiclass classification (up, down, no change)
K = 5 
LEARNING_RATE = 0.001
EPOCHS = 10
