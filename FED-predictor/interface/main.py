# Your main Python entry point containing all "routes"

import logging
import pandas as pd
from pathlib import Path

from params import *
from ml_logic.data import load_raw_data, ensure_dir_exists, adjust_column_names, format_raw_data, sort_dates, text_encode, group_text, sliding_window
from ml_logic.encoders import ordinal_encode, finalize_df
from ml_logic.model import train_model 
from ml_logic.registry import save_model, save_results


# Set up logging once at the beginning of the script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

######## load_and_format ########
# Description: 
# Args: 
# Kwargs: N/A
# Seps: 
# Output : pairing_df 

def load_and_format ():
        
    try: 
        # load_raw_data
        text_df, rates_df = load_raw_data()
    
        # adjust_column_names, rename columns in rates_df
        text_df = adjust_column_names(text_df)
        rates_df = adjust_column_names(rates_df, RENAME_RATES)
    
        # formats the relevant columns in the dataframes
        text_df, rates_df = format_raw_data(text_df, rates_df)
    
        # sort_dates
        text_df = sort_dates(text_df, 'text')
        rates_df = sort_dates(rates_df, 'rates', reference_df=text_df)
    
        # text_encoder
        text_df = text_encode(text_df, 'type', 'text')
        rates_df = text_encode(rates_df, 'rate', 'rates')
    
        # sliding_window
        pairing_df = sliding_window(rates_df, text_df)
    
        # Save processed data
        output_path = RAW_DATA_PATH
        ensure_dir_exists(output_path)  # Ensure the directory exists
        pairing_df.to_csv(Path(output_path) / "formatted_df.csv", index=False)
        
        logger.info("Data loaded and formatted successfully.")
        return pairing_df
    
    except Exception as e:
        logger.error(f"Error in load_and_format: {e}")
        raise
   
    
######## preprocess ########
# Description: process the final paired dataframe
# Args: 
# Kwargs: N/A
# Seps:  Apply ordinal encoding to the decision column.
#        Vectorize the grouped texts using FinBERT and combine embeddings.
#        Split into train and test sets.
#        Convert train and test arrays to PyTorch tensors.
# Output : X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor

def preprocess ():

    # Load data
    input_path = RAW_DATA_PATH
    pairing_df = pairing_df = pd.read_csv(Path(input_path) / "formatted_df.csv")

    # ordional_encode the rate decision
    pairing_df = ordinal_encode(pairing_df, 'decision')

    # finalize_df
    pairing_df = finalize_df(pairing_df)
        
    # Save processed data
    output_path = PROCESSED_DATA_PATH
    ensure_dir_exists(output_path)  # Ensure the directory exists
    pairing_df.to_csv(Path(output_path) / "preprocessed_df.csv", index=False)

    return pairing_df

    
######## train_and_evaluate ########
# Description: 
# Args: 
# Kwargs: N/A
# Seps:  
# Output : 

def train_and_evaluate():
    
    try:
        # Load data
        input_path = PROCESSED_DATA_PATH
        pairing_df = pairing_df = pd.read_csv(Path(input_path) / "preprocessed_df.csv")

        # initialize, compile, train and evaluate model
        model, average_accuracy = train_model(pairing_df)
    
        # Save the trained model and results
        save_model(model)
        save_results(average_accuracy)
        
        logger.info("Training completed and model saved successfully.")
        return model, average_accuracy  # Optionally return results if needed later
    
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {e}")
        raise

    return model, average_accuracy
