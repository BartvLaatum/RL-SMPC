import pandas as pd
import os

# Define parameters
horizons = range(1, 7)  # 1 to 6
uncertainty_values = [0.05, 0.1]
base_path = "data/SMPC/stochastic/smpc"

# Process each combination
for value in uncertainty_values:
    for h in horizons:
        # Construct filename
        filename = f"smpc-bounded-states-{h}H-10Ns-{value}.csv"
        filepath = os.path.join(base_path, filename)
        
        try:
            # Load the CSV
            df = pd.read_csv(filepath)
            
            # Update econ_rewards by adding penalties
            df['econ_rewards'] = df['econ_rewards'] + df['penalties']
            
            # Save back to the same file
            df.to_csv(filepath, index=False)
            print(f"Successfully processed {filename}")
            
        except FileNotFoundError:
            print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")