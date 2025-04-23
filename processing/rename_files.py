import os
import glob

# Directory containing the files
data_dir = "data/uncertainty-comparison/stochastic/mpc"

# Get list of all csv files in directory
files = glob.glob(os.path.join(data_dir, "*.csv"))

# Process each file
for file_path in files:
    # Get filename without directory
    filename = os.path.basename(file_path)
    
    # Check if file matches pattern we want to rename
    if filename.startswith("mpc-noise-correction-"):
        # Get new filename by removing "noise-correction-" part
        new_filename = filename.replace("mpc-noise-correction-", "mpc-")
        
        # Create full paths
        new_path = os.path.join(data_dir, new_filename)
        
        # Rename file
        os.rename(file_path, new_path)
        print(f"Renamed {filename} to {new_filename}")
