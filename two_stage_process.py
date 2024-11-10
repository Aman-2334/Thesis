import os
import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
import pandas as pd
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from thesis import MiOModel, train_model

model_save_path = "mio_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

class ComponentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=10):  # Adjust num_classes as needed
        super(ComponentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Step 1: Train and Save MiO Model on ASVspoof 2019
def train_and_save_mio_model():
    if not os.path.exists(model_save_path):
        train_model()
        print("MiO model trained and saved.")

# Step 2: Load Frozen MiO Model for Feature Extraction
def load_frozen_mio_model():
    mio_model = MiOModel().to(device)
    mio_model.load_state_dict(torch.load("mio_model_binary_classification.pth"))
    for param in mio_model.parameters():
        param.requires_grad = False  # Freeze all parameters
    return mio_model

def extract_features_with_mio(audio_path, mio_model):
    # Load audio file using soundfile
    waveform, sample_rate = sf.read(audio_path)

    # Ensure the waveform is in a PyTorch tensor format and has the correct dimensions
    waveform = torch.tensor(waveform, dtype=torch.float16).unsqueeze(0)  # Add batch dimension

    # Ensure sample rate compatibility (assuming MiO was trained with 16kHz)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Move waveform to device
    waveform = waveform.to(device)

    with torch.no_grad():
        # Pass the waveform directly through the frozen MiO model to get the feature embedding
        feature_tensor = mio_model(waveform)

    # Move feature tensor to CPU if needed
    feature_tensor = feature_tensor.cpu()

    return feature_tensor


class MLAADDataset(Dataset):
    def __init__(self, meta_file, root_dir, mio_model, label_type="acoustic"):
        self.data = pd.read_csv(meta_file)
        self.root_dir = root_dir
        self.mio_model = mio_model  # Pass the frozen MiO model to the dataset
        self.label_type = label_type  # "acoustic" or "vocoder"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.root_dir, row['file_path'])
        label = row[self.label_type]  # Either 'acoustic' or 'vocoder'

        # Extract feature embeddings using the frozen MiO model
        feature_tensor = extract_features_with_mio(file_path, self.mio_model)
        
        return feature_tensor, torch.tensor(label, dtype=torch.long)

# Stage 2: Train Lightweight Classification Head for Component Classification
def train_classification_head(classifier, train_loader, eval_loader, num_epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.001)

    # Train the classifier head
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate the classifier
    classifier.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in eval_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate and print evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Eval Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

mlaad_root_dir = "Dataset\mlaad\MLAADv3\\fake"  # Update with your actual MLAAD dataset path
combined_df = load_all_simpler_meta_files(mlaad_root_dir)

X = combined_df[['file_path']]  # Feature (file paths)
y = combined_df[['acoustic', 'vocoder']]  # Target columns (for stratified split)

train_df, eval_df = train_test_split(combined_df, test_size=0.2, stratify=y, random_state=42)
print("Training and evaluation splits created and saved.")

num_acoustic_classes = len(combined_df.acoustic.unique())
num_vocoder_classes = len(combined_df.vocoder.unique())

# Define two separate classifiers for acoustic and vocoder tasks
acoustic_classifier = ComponentClassifier(input_dim=120, num_classes=num_acoustic_classes).to(device)
vocoder_classifier = ComponentClassifier(input_dim=120, num_classes=num_vocoder_classes).to(device)

# Separate DataLoaders for Acoustic and Vocoder Classification Tasks
train_loader_acoustic = DataLoader(
    MLAADDataset("path_to_mlaad/train_meta.csv", mlaad_root_dir, label_type="acoustic"),
    batch_size=16, shuffle=True
)
eval_loader_acoustic = DataLoader(
    MLAADDataset("path_to_mlaad/eval_meta.csv", mlaad_root_dir, label_type="acoustic"),
    batch_size=16, shuffle=False
)

train_loader_vocoder = DataLoader(
    MLAADDataset("path_to_mlaad/train_meta.csv", mlaad_root_dir, label_type="vocoder"),
    batch_size=16, shuffle=True
)
eval_loader_vocoder = DataLoader(
    MLAADDataset("path_to_mlaad/eval_meta.csv", mlaad_root_dir, label_type="vocoder"),
    batch_size=16, shuffle=False
)

# Train and evaluate the acoustic classifier
print("Training Acoustic Classifier:")
train_classification_head(acoustic_classifier, train_loader_acoustic, eval_loader_acoustic)

# Train and evaluate the vocoder classifier
print("Training Vocoder Classifier:")
train_classification_head(vocoder_classifier, train_loader_vocoder, eval_loader_vocoder)
# Load and run the two-stage process
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
train_root = "path_to_MLAAD/train_audio"  # Root for training audio files
eval_root = "path_to_MLAAD/eval_audio"  # Root for eval audio files

# Train MiO on ASVspoof 2019
train_and_save_mio_model()

# Load frozen MiO as feature extractor
mio_model = load_frozen_mio_model()

# Prepare dataloaders for MLAAD train/eval
train_loader = DataLoader(MLAADDataset("path_to_mlaad/train_meta.csv", train_root), batch_size=16, shuffle=True)
eval_loader = DataLoader(MLAADDataset("path_to_mlaad/eval_meta.csv", eval_root), batch_size=16, shuffle=False)

# Train classifier head on MLAAD
train_classification_head(mio_model, train_loader, eval_loader)
