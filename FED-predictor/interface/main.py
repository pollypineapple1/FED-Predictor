# Your main Python entry point containing all "routes"

import logging
import pandas as pd
from pathlib import Path
import numpy as np
import torch
import pickle
from colorama import Fore, Style
import sys
import os

# Add the parent directory (FED-Predictor) to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from params import *

from ml_logic.data import load_raw_data, ensure_dir_exists, adjust_column_names, format_raw_data, sort_dates, text_encode, group_text, sliding_window
from ml_logic.encoders import ordinal_encode, finalize_df, prepare_input_text
from ml_logic.model import train_model 
from ml_logic.registry import save_model, save_results, load_model


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
    
        # Save processed data as pickle
        output_path = RAW_DATA_PATH
        ensure_dir_exists(output_path)  # Ensure the directory exists
        with open(Path(output_path) / "formatted_df.pkl", 'wb') as f:
            pickle.dump(pairing_df, f)
        
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

def preprocess():
    try:
        # Load data
        input_path = RAW_DATA_PATH
        with open(Path(input_path) / "formatted_df.pkl", 'rb') as f:
            pairing_df = pickle.load(f)

        # Ordinal encode the rate decision
        pairing_df = ordinal_encode(pairing_df, 'decision')

        # Finalize the DataFrame
        pairing_df = finalize_df(pairing_df)
        pairing_df['combined_vectorization'] = np.array(pairing_df['combined_vectorization'])

        # Save processed data as pickle
        output_path = PROCESSED_DATA_PATH
        ensure_dir_exists(output_path)  # Ensure the directory exists
        with open(Path(output_path) / "preprocessed_df.pkl", 'wb') as f:
            pickle.dump(pairing_df, f)

        return pairing_df
    
    except Exception as e:
        logger.error(f"Error in preprocess: {e}")
        raise

    
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
        with open(Path(input_path) / "preprocessed_df.pkl", 'rb') as f:
            pairing_df = pickle.load(f)

        # Initialize, compile, train, and evaluate model
        model, average_accuracy, fold_accuracies = train_model(pairing_df)

        # Save the trained model and get the model path
        model_path = save_model(model)

        # Prepare evaluation parameters (you can add more details as needed)
        params = {
            "model_architecture": "YourModelArchitecture",  # Replace with actual architecture info
            "learning_rate": LEARNING_RATE,
            "batch_size": 32,  # Replace with your actual batch size
            "epochs": EPOCHS
        }

        # Save results (model parameters and metrics)
        save_results(params, {"accuracy": average_accuracy, "average_accuracy": average_accuracy}, model_path)


        logger.info("Training completed and model saved successfully.")
        return model, average_accuracy  # Optionally return results if needed later
    
    except Exception as e:
        logger.error(f"Error in train_and_evaluate: {e}")
        raise
    

######## predict_test ########
# Description: 
# Args: 
# Kwargs: N/A
# Seps:  
# Output :  
    
def predict_test():
    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)
    
    # # Construct full file path and import the
    test_path = os.path.join(TEST_TEXT_PATH, TEST_FILE)
    
    # Prepare the input
    text_tensor  = prepare_input_text(test_path)

    # Load the trained model
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")  # Default to best model path
    model = load_model(model_path)  # Load your trained model from the saved checkpoint

    if not model:
        raise ValueError("No trained model found. Please train the model first.")

    model.eval()  # Set the model to evaluation mode

    # Make the prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(text_tensor)  # Forward pass

    # Get predicted class with highest probability
    prediction = torch.softmax(outputs, dim=1).numpy()  # Convert logits to probabilities
    predicted_class_index = np.argmax(prediction)

    # Class labels (you can adjust based on your model's output)
    class_names = ["up", "no change", "down"]

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]

    print('✅ Prediction complete.')
    print(f"Predicted Class: {predicted_class_label}")

    return predicted_class_label


######## predict ########
# Description: 
# Args: 
# Kwargs: N/A
# Seps:  
# Output :  
    
def predict(file):
    print(Fore.MAGENTA + "\n⭐️ Use case: predict" + Style.RESET_ALL)
       
    # Prepare the input
    text_tensor  = prepare_input_text(file)

    # Load the trained model
    model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")  # Default to best model path
    model = load_model(model_path)  # Load your trained model from the saved checkpoint

    if not model:
        raise ValueError("No trained model found. Please train the model first.")

    model.eval()  # Set the model to evaluation mode

    # Make the prediction
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model(text_tensor)  # Forward pass

    # Get predicted class with highest probability
    prediction = torch.softmax(outputs, dim=1).numpy()  # Convert logits to probabilities
    predicted_class_index = np.argmax(prediction)

    # Class labels (you can adjust based on your model's output)
    class_names = ["up", "no change", "down"]

    # Get the predicted class label
    predicted_class_label = class_names[predicted_class_index]

    print('✅ Prediction complete.')
    print(f"Predicted Class: {predicted_class_label}")

    return predicted_class_label