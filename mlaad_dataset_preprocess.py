import os
import pandas as pd

# Set the root directory of the MLAAD dataset
root_dir = "path_to_MLAAD_dataset/fake"  # Replace with the actual path to the 'fake' folder

# Initialize an empty set to store unique model names
unique_models = set()

# Function to process each 'simpler_meta.csv' file and collect unique models
def collect_unique_models(model_dir):
    # Path to 'simpler_meta.csv' file
    simpler_meta_path = os.path.join(model_dir, 'simpler_meta.csv')
    
    # Check if 'simpler_meta.csv' exists
    if os.path.isfile(simpler_meta_path):
        # Load simpler_meta.csv into a DataFrame
        simpler_meta_df = pd.read_csv(simpler_meta_path)
        
        # Collect unique models from the 'model' column
        unique_models.update(simpler_meta_df['model'].unique())
    else:
        print(f"'simpler_meta.csv' not found in {model_dir}")

# Function to process each 'model_L_K' directory
def process_model_directory(model_dir):
    # Path to the existing 'meta.csv' file
    meta_file_path = os.path.join(model_dir, 'meta.csv')
    
    # Check if 'meta.csv' exists
    if os.path.isfile(meta_file_path):
        # Load meta.csv into a DataFrame
        meta_df = pd.read_csv(meta_file_path)
        
        # Create the simpler meta DataFrame
        simpler_meta_df = meta_df[['path', 'model_name', 'architecture']].copy()
        simpler_meta_df.columns = ['file_path', 'vocoder', 'acoustic']  # Rename columns
        
        # Path to save the simpler meta file
        simpler_meta_path = os.path.join(model_dir, 'simpler_meta.csv')
        
        # Save the simpler meta DataFrame to CSV
        simpler_meta_df.to_csv(simpler_meta_path, index=False)
        print(f"Simpler metadata saved to: {simpler_meta_path}")
    else:
        print(f"'meta.csv' not found in {model_dir}")

# Traverse the directory structure
for language_dir in os.listdir(root_dir):
    language_path = os.path.join(root_dir, language_dir)
    if os.path.isdir(language_path):
        for model_dir in os.listdir(language_path):
            model_path = os.path.join(language_path, model_dir)
            if os.path.isdir(model_path):
                # Process each model directory to create simpler_meta.csv
                process_model_directory(model_path)
                # Collect unique models from each model directory
                collect_unique_models(model_path)
                

# Convert the set to a sorted list
unique_model_list = sorted(unique_models)

# Save the unique model list to a file
output_file = "unique_models.txt"
with open(output_file, "w") as f:
    for model in unique_model_list:
        f.write(model + "\n")

print("Unique models have been saved to:", output_file)
