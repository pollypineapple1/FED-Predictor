# Save and load models

from tensorflow import keras
import torch
import os
import time
import pickle
from params import *
from ml_logic.data import ensure_dir_exists

def save_model(model: torch.nn.Module, model_dir: str = LOCAL_REGISTRY_PATH) -> None:
    """
    Save the trained PyTorch model to disk using model's state_dict.
    The model is saved to a path including a timestamp, like:
    f"{model_dir}/models/{timestamp}.pth"
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, "models", f"{timestamp}.pth")
    
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(model_path))
    
    # Save the model's state_dict (recommended way to save PyTorch models)
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")
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
    # Correct file path with timestamp + .pickle
    results_path = os.path.join(LOCAL_REGISTRY_PATH, "params", f"{timestamp}.pickle")

    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(results_path))

    # Save the results to the file
    with open(results_path, "wb") as f:
        pickle.dump({"params": params, "metrics": metrics}, f)

    print(f"Results saved to {results_path}")