import pandas as pd

from data import load_raw_data

def test_load_data():

    text_df, rates_df = load_raw_data()

    # Check if dataframes were loaded successfully and print them
    if text_df is not None and rates_df is not None:
        # Print the first 5 rows of each dataframe
        print("Fed Scrape Data (first 5 rows):")
        print(text_df.head())

        print("\nUS Fed Rate Data (first 5 rows):")
        print(rates_df.head())
    else:
        print("There was an error loading the data.")

    print(f"Predicted Class: {predicted_class_label}")            