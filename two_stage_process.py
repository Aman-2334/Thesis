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
from whisper_feature_extractor import CNNFeatureExtractor, whisper_processor, whisper_model
from xlsr_feature_extractor import CNNFeatureExtractor, xlsr_model

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
    mio_model.load_state_dict(torch.load(model_save_path))
    for param in mio_model.parameters():
        param.requires_grad = False  # Freeze all parameters
    return mio_model

def preprocess_audio_with_whisper_xlsr(audio_path):
    # Load audio
    waveform, sample_rate = sf.read(audio_path)
    waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)

    # Resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)

    waveform_cpu = waveform.cpu().numpy()
    # Extract Whisper CNN features
    with torch.no_grad():
        mel_features = whisper_processor.feature_extractor(
            waveform_cpu,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.squeeze(0)  # Remove batch dimension for single sample

        # Pad or truncate mel-spectrogram to TARGET_MEL_LENGTH
        if mel_features.shape[1] < 3000:
            padding = 3000 - mel_features.shape[1]
            mel_features = torch.nn.functional.pad(mel_features, (0, padding), mode='constant', value=0)
        else:
            mel_features = mel_features[:, :3000]

        # Now pass through Whisper
        mel_features = mel_features.to(device)
        whisper_encoder_outputs = whisper_model.encoder(mel_features.unsqueeze(0))
        cnn_extractor = CNNFeatureExtractor(input_dim=512, output_dim=120).to(device)
        cnn_features_whisper = cnn_extractor(whisper_encoder_outputs.last_hidden_state)

    # Extract XLS-R CNN features
    with torch.no_grad():
        xlsr_features = xlsr_model(waveform)
        cnn_extractor = CNNFeatureExtractor(input_dim=1280, output_dim=120).to(device)
        cnn_features_xlsr = cnn_extractor(xlsr_features.last_hidden_state)
    
    return cnn_features_whisper, cnn_features_xlsr

def extract_features_with_mio(audio_path, mio_model):
    # Get the CNN features from Whisper and XLS-R
    cnn_features_whisper, cnn_features_xlsr = preprocess_audio_with_whisper_xlsr(audio_path)

    # Pass through the frozen MiO model to get the feature embedding
    with torch.no_grad():
        feature_tensor = mio_model(cnn_features_whisper, cnn_features_xlsr, return_embedding=True)

    # Move feature tensor to CPU if needed
    feature_tensor = feature_tensor.cpu().detach()
    # print("feature_tensor",feature_tensor.shape)
    return feature_tensor


class MLAADDataset(Dataset):
    def __init__(self, meta_df, root_dir, mio_model, label_type="acoustic"):
        self.data = pd.DataFrame(meta_df)
        self.root_dir = root_dir
        self.mio_model = mio_model  # Pass the frozen MiO model to the dataset
        self.label_type = label_type  # "acoustic" or "vocoder"
        self.label_mapping = acoustic_label_mapping if label_type == "acoustic" else vocoder_label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = os.path.join(self.root_dir, row['file_path'])
        file_path = file_path.replace("./fake/","").replace("/","\\");
        label_str = row[self.label_type]  # Either 'acoustic' or 'vocoder'
        
        # Map string label to integer
        label = self.label_mapping[label_str]

        # Extract feature embeddings using the frozen MiO model
        feature_tensor = extract_features_with_mio(file_path, self.mio_model)
        
        return feature_tensor, torch.tensor(label, dtype=torch.long)

# Stage 2: Train Lightweight Classification Head for Component Classification
def train_classification_head(classifier, train_loader, num_epochs=10, learning_rate=0.001, verbose=True):
    """
    Train a classifier head on provided data.
    
    Args:
        classifier (nn.Module): The classifier model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        num_epochs (int): Number of epochs to train.
        learning_rate (float): Learning rate for the optimizer.
        verbose (bool): If True, prints detailed logs during training.
        
    Returns:
        nn.Module: The trained classifier model.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        for batch_idx, (features, labels) in enumerate(train_loader):
            # Move data to device
            features, labels = features.to(device), labels.to(device)
            # print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = classifier(features).squeeze(1)
            # print(f"Classifier outputs shape: {outputs.shape}")
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Optionally print batch-level details
            if verbose and batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Print epoch-level loss
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {running_loss/len(train_loader):.4f}")

    print("Training completed.")
    return classifier

def evaluate_classification_head(classifier, eval_loader):
    """
    Evaluate the trained classifier head on provided evaluation data.
    
    Args:
        classifier (nn.Module): The trained classifier model.
        eval_loader (DataLoader): DataLoader for evaluation data.
        
    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    classifier.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in eval_loader:
            # Move data to device
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = classifier(features).squeeze(1)
            _, preds = torch.max(outputs, 1)
            
            # Collect predictions and true labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    results = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    
    # Print results
    print(f"Evaluation -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    
    return results


mlaad_root_dir = "Dataset\mlaad\MLAADv3\\fake"  # Update with your actual MLAAD dataset path
combined_df = load_all_simpler_meta_files(mlaad_root_dir)

X = combined_df[['file_path']]  # Feature (file paths)
y = combined_df[['acoustic', 'vocoder']]  # Target columns (for stratified split)

train_df, eval_df = train_test_split(combined_df, test_size=0.2, stratify=y, random_state=42)
print("Training and evaluation splits created and saved.")

num_acoustic_classes = len(combined_df.acoustic.unique())
num_vocoder_classes = len(combined_df.vocoder.unique())

unique_acoustic_labels = combined_df['acoustic'].unique()
unique_vocoder_labels = combined_df['vocoder'].unique()

acoustic_label_mapping = {label: idx for idx, label in enumerate(unique_acoustic_labels)}
vocoder_label_mapping = {label: idx for idx, label in enumerate(unique_vocoder_labels)}
# Define two separate classifiers for acoustic and vocoder tasks
acoustic_classifier = ComponentClassifier(input_dim=120, num_classes=num_acoustic_classes).to(device)
vocoder_classifier = ComponentClassifier(input_dim=120, num_classes=num_vocoder_classes).to(device)
train_and_save_mio_model()
mio_model = load_frozen_mio_model()
# Separate DataLoaders for Acoustic and Vocoder Classification Tasks
train_loader_acoustic = DataLoader(
    MLAADDataset(train_df, mlaad_root_dir, mio_model, label_type="acoustic"),
    batch_size=16, shuffle=True
)
eval_loader_acoustic = DataLoader(
    MLAADDataset(eval_df, mlaad_root_dir, mio_model, label_type="acoustic"),
    batch_size=16, shuffle=False
)

train_loader_vocoder = DataLoader(
    MLAADDataset(train_df, mlaad_root_dir, mio_model, label_type="vocoder"),
    batch_size=16, shuffle=True
)
eval_loader_vocoder = DataLoader(
    MLAADDataset(eval_df, mlaad_root_dir, mio_model, label_type="vocoder"),
    batch_size=16, shuffle=False
)

# Train and evaluate the acoustic classifier
print("Training Acoustic Classifier:")
acoustic_classifier = train_classification_head(acoustic_classifier, train_loader_acoustic)

print("Evaluating Acoustic Classifier:")
evaluate_classification_head(acoustic_classifier, eval_loader_acoustic)

print("Training Vocoder Classifier:")
vocoder_classifier = train_classification_head(vocoder_classifier, train_loader_vocoder)

print("Evaluating Vocoder Classifier:")
evaluate_classification_head(vocoder_classifier, eval_loader_vocoder)