# Save and load models

from tensorflow import keras
import torch
import os
import time
import pickle
import shutil

from params import *
from ml_logic.data import ensure_dir_exists
from ml_logic.model import BiLSTM

def save_model(model: torch.nn.Module, model_dir: str = LOCAL_REGISTRY_PATH) -> str:
    """
    Save the trained PyTorch model to disk using model's state_dict.
    Also saves a timestamp for tracking purposes.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(model_dir, "models", f"{timestamp}.pth")
    
    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(model_path))
    
    # Save the model's state_dict (recommended way to save PyTorch models)
    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")
    return model_path


def load_model(model_path: str = None, input_dim=IMPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
    """
    Load the BiLSTM model from a checkpoint and initialize it with the correct architecture.
    """
    if model_path is None:
        model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        print(f"Attempting to load best model from: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))  # Load the saved state dictionary

        # 🔹 Ensure input_dim is provided (# features)
        if input_dim is None:
            raise ValueError("`input_dim` must be provided to initialize the model.")

        # 🔹 Initialize the BiLSTM model with correct dimensions
        model = BiLSTM(input_dim, hidden_dim, output_dim)

        # 🔹 Load the saved state dictionary
        model.load_state_dict(checkpoint)

        # 🔹 Set the model to evaluation mode
        model.eval()
        print("✅ BiLSTM model loaded successfully and set to evaluation mode.")

        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise OSError(f"Unable to load model from '{model_path}': {e}")


def save_results(params: dict, metrics: dict, model_path: str) -> None:
    """
    Save params & metrics to disk and track the best model.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Correct file path with timestamp + .pickle
    results_path = os.path.join(LOCAL_REGISTRY_PATH, "params", f"{timestamp}.pickle")

    # Ensure the directory exists
    ensure_dir_exists(os.path.dirname(results_path))

    # Save the results to the file
    with open(results_path, "wb") as f:
        pickle.dump({"params": params, "metrics": metrics, "model_path": model_path}, f)

    print(f"Results saved to {results_path}")

    # Track best model
    save_best_model()


def save_best_model() -> None:
    """
    Find the best model based on highest accuracy and save it as 'best_model.pth' in the checkpoint directory.
    """
    params_dir = os.path.join(LOCAL_REGISTRY_PATH, "params")
    models_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")

    if not os.path.exists(params_dir):
        print("No parameters directory found, skipping best model update.")
        return

    best_model_path = None
    best_accuracy = 0.0

    for filename in os.listdir(params_dir):
        if filename.endswith(".pickle"):
            params_path = os.path.join(params_dir, filename)

            with open(params_path, "rb") as f:
                data = pickle.load(f)

            # Ensure metrics and accuracy exist
            if "metrics" in data and "accuracy" in data["metrics"] and "model_path" in data:
                accuracy = data["metrics"]["accuracy"]
                model_path = data["model_path"]

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_path = model_path

    if best_model_path and os.path.exists(best_model_path):
        ensure_dir_exists(CHECKPOINT_DIR)  # Ensure the directory exists
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)  # ✅ Add this line
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
        shutil.copy(best_model_path, checkpoint_path)
        print(f"Best model updated: {checkpoint_path} (Accuracy: {best_accuracy:.4f})")
    else:
        print("No valid best model found.")
    
