# Custom encoder utilities

# IMPORTS
import numpy as np
import pandas as pd
from datetime import timedelta

import torch
from sklearn.preprocessing import OrdinalEncoder
from transformers import AutoTokenizer, AutoModel, pipeline


######## adjust_column_names ########
# Description: simplifies column names
# Args: dataframe to act on
# Kwargs: dictionary of new desired pairings 
# Seps: converts all column names to small
#       changes the names according to provided lists
# Output : one dataframe with simplified column names

def adjust_column_names(df, rename_dict=None):
    
    df.columns = df.columns.str.lower()
    
    if rename_dict:  # Only rename if a valid dictionary is provided
        df = df.rename(columns=rename_dict)

    return df

######## format_raw_data ########
# Description: formats raw data
# Args: dataframe and columns
# Kwargs: N/A
# Seps: convert date columns to datetime
#       convert rate to % floats
#       filters rates_df to match text_df range
# Output : tuple with two dataframes  

def format_raw_data(
    text_df, rates_df, 
    text_date_col='date', 
    rates_date_col='date', 
    rate_col='rate'
):

    text_df[text_date_col] = pd.to_datetime(text_df[text_date_col])
    rates_df[rates_date_col] = pd.to_datetime(rates_df[rates_date_col], format='%b %d, %Y')

    rates_df[rate_col] = rates_df[rate_col].str.rstrip('%').astype(float)

    start_date_text_df = text_df[text_date_col].min()
    rates_df = rates_df[rates_df[rates_date_col] >= start_date_text_df]

    return text_df, rates_df

######## sort_dates ########
# Description: sorts row data by date, and deletes old rates decisions
# Args: df to act on, type of df, reference df
# Kwargs: N/A
# Seps: sets date columnd as index
#       sets minimum start date to be applied to rates and filters the dataframe
# Output : tuple with two dataframes 

def sort_dates(df, df_type, reference_df=None):
    
    # Ensure 'date' is a column (if already an index, reset it)
    if 'date' in df.index:
        df = df.reset_index()
    
    df = df.sort_values(by='date')  # Sort by date
    
    # If df_type is 'rates', filter based on reference_df (text_df)
    if df_type == 'rates' and reference_df is not None:
        start_date = reference_df['date'].min()  # Get earliest date from reference_df
        df = df[df['date'] >= start_date]  # Filter df where date is >= start_date

    return df

######## text_encode ########
# Description: text-encodes the type of text or rate decision
# Args: dataframe, column name, dataframe keyword
# Kwargs: 
# Seps: with a conditionality based on dataframe keyword
#       one hot encode type into type_text if text_df
#       calculates difference and one hots encode rate_change into rate_change_text if rates_df#       
# Output: dataframe
   
def text_encode(df, column, df_type):
    
    if df_type == 'text':  # Check if it's a text DataFrame
        df['type_text'] = df[column].apply(lambda x: 'statement' if x == 0 else 'minutes')

    elif df_type == 'rates':  # Check if it's a rates DataFrame
        df['rate_change'] = df[column].diff()
        df['rate_change_text'] = df['rate_change'].apply(lambda x: 'up' if x > 0 else ('down' if x < 0 else 'no change')).astype(str)
        
    else:  # Handle incorrect df_type values
        return "Error: Invalid df_type. Choose 'text' or 'rates'."

    return df

######## group_text ########
# Description: function used within sliding_window to group text based on a time window size
# Args: dataframes and 
# Kwargs: 
# Seps:     
# Output: 

def group_text(rate_date, text_df, date_diff):
    window_size = timedelta(days=date_diff)

    # Filter texts that occurred before the rate decision
    valid_texts = text_df[text_df['date'] < rate_date]

    # Apply sliding window: Get texts within the specified window size before rate_date
    texts_in_window = valid_texts[valid_texts['date'] >= rate_date - window_size]

    # Combine the texts
    grouped_texts = ' '.join(texts_in_window['text'])
    
    return grouped_texts

######## sliding_window ########
# Description: allows to pair condensed text and rates decision in one dataframe 
#               while respective sequentiality in text
# Args: 
# Kwargs: 
# Seps:     
# Output: 

def sliding_window(rates_df, text_df):
    
    rates_df = rates_df.copy()
    
    # Calculate the difference between consecutive rate decisions to determine dynamic window size
    rates_df['next_date'] = rates_df['date'].shift(-1)
    rates_df['date_diff'] = (rates_df['next_date'] - rates_df['date']).dt.days
    
    # Corrects for NaNs since we are subtracting time deltas
    rates_df['date_diff'] = rates_df['rate_diff'].fillna(0).astype(int)

    # isolate statements and minutes as they occurr at different times relative to previous decisions
    statement_df = text_df[text_df['type_text'] == 'statement']
    minutes_df = text_df[text_df['type_text'] == 'minutes']    

    pairing_data  = []

    for _, rate_row in rates_df.iterrows():
        rate_date = rate_row['date']
        rate = rate_row['rate_change_text']
        date_diff = rate_row['date_diff']
        
        grouped_statements = group_text(rate_date, statement_df, date_diff)
        grouped_minutes = group_text(rate_date, minutes_df, date_diff)
    
        # Add the data to pairing_df
        pairing_data.append({
            'decision': rate,
            'date': rate_date,
            'grouped_statements': grouped_statements,
            'grouped_minutes': grouped_minutes,
            'window_size_days': date_diff  # Store the dynamic window size for reference
        })

    pairing_df = pd.DataFrame(pairing_data)
    pairing_df = pairing_df.set_index("date")
    
    return pairing_df


######## ordinal_encode ########
# Description: encodes the type of text or rate decision
# Args: dataframe, column name, dataframe keyword
# Kwargs: 
# Seps: with a conditionality based on dataframe keyword
#       one hot encode type into type_text if text_df
#       calculates difference and one hots encode rate_change into rate_change_text if rates_df#       
# Output: dataframe

def ordinal_encode(df, columns):
    
    ordinal_encoder = OrdinalEncoder()
    
    for column in columns:
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
        return np.zeros((768,))  # Return a zero-vector if input is empty or NaN
    
    # Proceed with tokenization and embedding generation
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state[:, 0, :].numpy()  

######## fianl_df ########
# Description:  prepares the final dataframe before modeling
# Args: 
# Kwargs: 
# Seps:        
# Output: dataframe

def final_df(df, columns):
    
    df['statement_vectorized'] = df['grouped_statements'].apply(lambda x: FinBERT_vectorizaion(str(x)))
    df['minutes_vectorized'] = df['grouped_minutes'].apply(lambda x: FinBERT_vectorizaion(str(x)))
    
    print(df).head()
    
    return df[columns]