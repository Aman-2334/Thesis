import os
import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from thesis import MiOModel

def load_all_simpler_meta_files(root_dir):
    all_data = []
    for language_dir in os.listdir(root_dir):
        language_path = os.path.join(root_dir, language_dir)
        if os.path.isdir(language_path):
            for model_dir in os.listdir(language_path):
                model_path = os.path.join(language_path, model_dir)
                simpler_meta_path = os.path.join(model_path, 'simpler_meta.csv')
                if os.path.isfile(simpler_meta_path):
                    try:
                        # Load the simpler_meta.csv and append to the list
                        df = pd.read_csv(simpler_meta_path)
                        all_data.append(df)
                    except pd.errors.EmptyDataError:
                        print(f"Skipping empty file: {simpler_meta_path}")
    # Concatenate all loaded data into a single DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total records loaded: {len(combined_df)}")
    return combined_df

# Step 1: Load all simpler_meta.csv files
root_dir = "Dataset\mlaad\MLAADv3\\fake"  # Update with your actual MLAAD dataset path
combined_df = load_all_simpler_meta_files(root_dir)

# Step 2: Perform stratified split
X = combined_df[['file_path']]  # Feature (file paths)
y = combined_df[['acoustic', 'vocoder']]  # Target columns (for stratified split)

train_df, eval_df = train_test_split(combined_df, test_size=0.2, stratify=y, random_state=42)

# Step 3: Save the splits to new CSV files
train_df.to_csv('mlaad_train_meta.csv', index=False)
eval_df.to_csv('mlaad_eval_meta.csv', index=False)

print("Training and evaluation splits created and saved.")

mio_model = MiOModel()  # Initialize model structure
mio_model.load_state_dict(torch.load("path/to/mio_model_binary_classification.pth"))
mio_model.eval()  # Set to evaluation mode

# Freeze MiO model parameters to prevent further training
for param in mio_model.parameters():
    param.requires_grad = False

# Define the lightweight classification head
class ComponentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):  # Adjust num_classes as per unique components
        super(ComponentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Combine MiO model and classification head
class TwoStageModel(nn.Module):
    def __init__(self, feature_extractor, classifier):
        super(TwoStageModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier

    def forward(self, whisper_features, xlsr_features):
        # Extract features using frozen MiO model
        extracted_features = self.feature_extractor(whisper_features, xlsr_features)
        # Pass extracted features through the classification head
        return self.classifier(extracted_features)

# Instantiate the classification head and combined model
input_dim = 120  # Adjust according to the output dimension of MiO feature extraction
num_classes = len(unique_components)  # Replace with actual number of unique components (models)
classifier_head = ComponentClassifier(input_dim=input_dim, num_classes=num_classes)
two_stage_model = TwoStageModel(mio_model, classifier_head).to(device)
