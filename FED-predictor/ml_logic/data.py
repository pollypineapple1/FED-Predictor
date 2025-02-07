# Save, load and clean data
# First set of functions to exectue

# IMPORTS
import pandas as pd
import os
from dotenv import load_dotenv
from colorama import Fore, Style
from datetime import timedelta
from pathlib import Path

from params import RAW_DATA_PATH, TEXT_FILE, RATES_FILE

# Helper function to ensure directory exists
def ensure_dir_exists(path):
    Path(path).mkdir(parents=True, exist_ok=True)

######## load_raw_data ########
# Description: import raw from a local folder into a dataframe
# Args: folder path
# Kwargs: N/A
# Seps: defines local folder path as variable
#       pull data using path variable
# Output : tuple with two dataframes 

def load_raw_data():

    # Load environment variables from .env file
    load_dotenv(override=True)
    
    print(Fore.MAGENTA + "\nLoading raw data..." + Style.RESET_ALL)
        
    # Construct full file paths
    text_data = os.path.join(RAW_DATA_PATH, TEXT_FILE)
    rates_data = os.path.join(RAW_DATA_PATH, RATES_FILE)

     # Debugging: Check if paths exist
    if not os.path.exists(text_data):
        raise FileNotFoundError(f"Error: The text file was not found at {text_data}")

    if not os.path.exists(rates_data):
        raise FileNotFoundError(f"Error: The rates file was not found at {rates_data}")

    
    text_df = pd.read_csv(text_data)
    rates_df = pd.read_csv(rates_data)
    
    print(f"Data loaded from {text_data, rates_data}")
    
    return text_df, rates_df


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
    date='date',
    rate='rate'
):

    text_df[date] = pd.to_datetime(text_df[date], format='%Y%m%d')
    rates_df[date] = pd.to_datetime(rates_df[date], format='%b %d, %Y')

    rates_df[rate] = rates_df[rate].str.rstrip('%').astype(float)

    start_date_text_df = text_df[date].min()
    rates_df = rates_df[rates_df[date] >= start_date_text_df]

    return text_df, rates_df

######## sort_dates ########
# Description: sorts row data by date, and deletes old rates decisions
# Args: df to act on, type of df (string), reference df
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
#       calculates difference and one hots encode rate_change into rate_change_text if rates_df      
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
    rates_df['date_diff'] = rates_df['date_diff'].fillna(0).astype(int)

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
