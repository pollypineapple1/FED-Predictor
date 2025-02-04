# Save, load and clean data
# First set of functions to exectue

# IMPORTS
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
from colorama import Fore, Style

######## load_raw_data ########
# Description: import raw from a local folder into a dataframe
# Args: folder path
# Kwargs: N/A
# Seps: defines local folder path as variable
#       pull data using path variable
# Output : tuple with two dataframes 

def load_raw_data():
    
    print(Fore.MAGENTA + "\nLoading raw data..." + Style.RESET_ALL)
    
    load_dotenv()
    
    base_path = os.getenv("BASE_PATH")
    data_dir = os.getenv("DATA_DIR")
    text_file = os.getenv("TEXT_FILE")
    rates_file = os.getenv("RATES_FILE")
    
    TEXT_DATA = os.path.join(base_path, data_dir, text_file)
    RATES_DATA = os.path.join(base_path, data_dir, rates_file)
    
    text_df = pd.read_csv(TEXT_DATA)
    rates_df = pd.read_csv(RATES_DATA)
    
    print(f"Data loaded from {TEXT_DATA, RATES_DATA}")
    
    return text_df, rates_df