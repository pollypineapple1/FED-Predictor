# Save and load models

######## fianl_df ########
# Description:  prepares the final dataframe before modeling
# Args: 
# Kwargs: 
# Seps:        
# Output: dataframe

def save_df(df, output_path):
    
    df.to_csv(output_path, index=False)
    
    print(f"Results saved to {output_path}")
