# Custom encoder utilities
# Second set of functions to execute

# IMPORTS
import numpy as np
import pandas as pd
import ast

import torch
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from params import TEST_SIZE, RANDOM_STATE

######## ordinal_encode ########
# Description: encodes the type of text or rate decision
# Args: dataframe, column name
# Kwargs: 
# Seps: encodes the text rate decision format into integer   
# # Output: dataframe

def ordinal_encode(df, column):
    
    ordinal_encoder = OrdinalEncoder()
    df[f'{column}_encoded'] = ordinal_encoder.fit_transform(df[[column]]).astype(int)
    
    return df

######## FinBERT_vectorizaion ########
# Description:  FinBERT is a pre-trained vectorizer that knows how to handle domain-specific raw text
# Args: 
# Kwargs: 
# Seps:        
# Output: dataframe

def FinBERT_vectorizaion(text):
    
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModel.from_pretrained("ProsusAI/finbert")
    
    # Ensure text is not empty or NaN
    if pd.isna(text) or text.strip() == "":
        return np.zeros((768,),dtype=np.float32)  # Return a zero-vector if input is empty or NaN
    
    # Proceed with tokenization and embedding generation
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy().astype(np.float32)  

######## fianl_df ########
# Description:  prepares the final dataframe before modeling
# Args: 
# Kwargs: 
# Seps:        
# Output: dataframe

def finalize_df(df):
    
    # Vectorize statements and minutes
    df['statement_vectorized'] = df['grouped_statements'].apply(lambda x: np.array(FinBERT_vectorizaion(str(x)), dtype=np.float32))
    df['minutes_vectorized'] = df['grouped_minutes'].apply(lambda x: np.array(FinBERT_vectorizaion(str(x)), dtype=np.float32))

    # Combine the two vectors into one
    df['combined_vectorization'] = df.apply(
        lambda row: np.hstack((
            row['statement_vectorized'],  # Ensure this is a numpy array
            row['minutes_vectorized'] if isinstance(row['minutes_vectorized'], np.ndarray) else np.zeros(768, dtype=np.float32)
        )),
        axis=1
    )
    
    # Make sure the 'combined_vectorization' column is stored as a numpy array of dtype float32
    df['combined_vectorization'] = df['combined_vectorization'].apply(lambda x: np.array(x, dtype=np.float32))
    
    # Debugging: Check data type after processing
    print(df['combined_vectorization'].apply(type).head(3))  # Should show <class 'numpy.ndarray'>
    
    return df

######## train_test_split ########
# Description:  splits variables into X and y
# Args: 
# Kwargs: 
# Seps:        
# Output: np.arrays

def custom_train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    
    # Convert stringified arrays back to real NumPy arrays
    df['combined_vectorization'] = df['combined_vectorization'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32) if isinstance(x, str) else x)

    # Stack the arrays to create the feature matrix
    X = np.vstack(df['combined_vectorization'].values)
    y = df['decision_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

######## class_weighting ########
# Description:  converts each class occurrence with a weight and stores it for use in model.py
# Args: 
# Kwargs: 
# Seps:        
# Output: weights of each class

def class_weighting (y_train):

    # Convert y_train to a Pandas Series to use value_counts
    y_train = pd.Series(y_train)
    
    # Count occurrences in y_train
    class_counts = y_train.value_counts().sort_index().values  # Ensure ordering is correct

    # Compute class weights
    class_weights = torch.tensor([1 / count for count in class_counts], dtype=torch.float32)
    
    return class_weights
######## tensor_conversion ########
# Description:  converts data to PyTorch tensors
# Args: 
# Kwargs: 
# Seps:        
# Output: tensors

def tensor_conversion(X_train, y_train):
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # BiLSTM will need 3D tensors. our tensors only have 2Ds for X
    # conversely, only 1D tensor for y with integer data, and not float
    X_train_tensor = X_train_tensor.unsqueeze(1)
    
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    return X_train_tensor, y_train_tensor