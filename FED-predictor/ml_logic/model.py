# Model related functions
import numpy as np

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import accuracy_score

from params import HIDDEN_DIM, OUTPUT_DIM, K, LEARNING_RATE, RANDOM_STATE, EPOCHS
from ml_logic.encoders import class_weighting, custom_train_test_split, tensor_conversion

# we build the model through a Class function because it's easier to separate tasks and use the mo`del mo`dularly
# we also store various statistics this way within class comands so we do not have to worry about `it`

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]  # Take the last time step's output
        return self.fc(self.relu(last_out))


######## initialize_model ########
# Description:  converts data to PyTorch tensors
# Args: dimensionality of the input features
#       dimensionality of the hidden layers
#       number of classes for classification
# Kwargs: 
# Seps:        
# Output: BiLSTM model

def initialize_model(X_train, hidden_dim = HIDDEN_DIM, output_dim = OUTPUT_DIM):
    
    input_dim = X_train.shape[1]
    
    model = BiLSTM(input_dim, hidden_dim, output_dim)
    
    return model


######## compile_model ########
# Description:  Compiles the model by defining the loss function and optimizer
# Args: model and learning_rate to use
# Kwargs: 
# Seps:        
# Output: loss function to use and optimizer

def compile_model(y_train, model, learning_rate=LEARNING_RATE):
    
    class_weights = class_weighting (y_train)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    return criterion, optimizer


######## evaluate_model ########
# Description:  Compiles the model by defining the loss function and optimizer
# Args: model and learning_rate to use
# Kwargs: 
# Seps:        
# Output: loss function to use and optimizer

def evaluate_model(model, X_test_tensor, y_test_tensor):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation for efficiency
        outputs = model(X_test_tensor)  # Get the model's output

        logits = outputs  # These are the raw predictions (logits)

        # Get the predicted class (highest logit)
        predicted_labels = torch.argmax(logits, dim=1)
        
    # Compute accuracy by comparing predicted labels to true labels
    accuracy = accuracy_score(y_test_tensor.numpy(), predicted_labels.numpy())
    
    return accuracy


######## train model_model ########
# Description:  train model using k-fold
# Args: 
# Kwargs: 
# Seps:        

def train_model(pairing_df, n_splits=K):
    
    X_train, X_test, y_train, y_test = custom_train_test_split(pairing_df)
    X_train_tensor, y_train_tensor = tensor_conversion(X_train,y_train)
    
    kf = KFold(n_splits=K, shuffle=True, random_state=RANDOM_STATE)
    
    fold_accuracies =[]
        
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_tensor, y_train_tensor)):
        print(f"Training fold {fold + 1}/{K}...")

        # Split data into training and validation sets for this fold
        X_train_fold, X_val_fold = X_train_tensor[train_idx], X_train_tensor[val_idx]
        y_train_fold, y_val_fold = y_train_tensor[train_idx], y_train_tensor[val_idx]

        # Convert to PyTorch tensors
        X_train_fold_tensor = torch.tensor(X_train_fold, dtype=torch.float32)
        y_train_fold_tensor = torch.tensor(y_train_fold, dtype=torch.long)
        X_val_fold_tensor = torch.tensor(X_val_fold, dtype=torch.float32)
        y_val_fold_tensor = torch.tensor(y_val_fold, dtype=torch.long)

        # Initialize the model, criterion, and optimizer
        model = initialize_model(X_train, hidden_dim=128, output_dim=3)
        criterion, optimizer = compile_model(y_train, model, learning_rate = LEARNING_RATE)

        # Train the model on this fold
        model.train()  # Set model to training mode
        for epoch in range(EPOCHS):
            optimizer.zero_grad()
            outputs = model(X_train_fold_tensor)
            loss = criterion(outputs, y_train_fold_tensor)
            loss.backward()
            optimizer.step()
            
        # Evaluate the model on the validation set
        accuracy = evaluate_model(model, X_val_fold_tensor, y_val_fold_tensor)
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")

        # Save the accuracy for this fold
        fold_accuracies.append(accuracy)

        # Calculate average accuracy across all folds
        average_accuracy = np.mean(fold_accuracies)
        print(f"\nAverage Cross-Validation Accuracy: {average_accuracy:.4f}")

    return model, average_accuracy

