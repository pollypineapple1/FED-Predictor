# Save and load models

from tensorflow import keras
import os
import time
import pickle
from params import *
from ml_logic.data import ensure_dir_exists

def save_model(model: keras.Model = None) -> None:
    """
    Persist trained model locally on the hard drive at f"{LOCAL_REGISTRY_PATH}/models/{timestamp}.h5"
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save model locally
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.keras")
    ensure_dir_exists(model_path)
    model.save(model_path)

    return None


def load_model(model_path=None) -> keras.Model:
    """
    Load the model from the specified path.
    Default path is within LOCAL_REGISTRY_PATH under models directory.
    """
    if model_path is None:
        model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", "best_model.keras")
        print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"No such file or directory: '{model_path}'")
        raise FileNotFoundError(f"No such file or directory: '{model_path}'")

    try:
        model = keras.models.load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise OSError(f"Unable to load model from '{model_path}': {e}")


def save_results(params: dict, metrics: dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/params/{current_timestamp}.pickle"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    results_path = os.path.join(LOCAL_REGISTRY_PATH, "params", f"{timestamp}.pickle")
    ensure_dir_exists(results_path)  # Ensure the directory exists
    
    with open(results_path, "wb") as f:
        pickle.dump({"params": params, "metrics": metrics}, f)

    print(f"Results saved to {results_path}")