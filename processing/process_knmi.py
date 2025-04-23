import pandas as pd
import glob
import os

# Define input and output directories
input_dir = "weather/train"
output_dir = "weather/train/processed"
os.makedirs(output_dir, exist_ok=True)

# Define the required column order
required_columns = ['time', 'global radiation', 'air temperature','RH', '??',  'CO2 concentration']

# Process each CSV file in the input directory
for file_path in glob.glob(os.path.join(input_dir, "*.csv")):
    # Load CSV data
    df = pd.read_csv(file_path)
    
    # Rearrange and filter the columns; any extra columns are dropped
    df_subset = df[required_columns]
    
    # Save the CSV without header
    file_name = os.path.basename(file_path)
    output_path = os.path.join(output_dir, file_name)
    df_subset.to_csv(output_path, index=False, header=False)