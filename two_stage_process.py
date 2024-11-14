import os
import torch
import pandas as pd
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchaudio
from whisper_feature_extractor import CNNFeatureExtractor as WhisperCNN
from xlsr_feature_extractor import CNNFeatureExtractor as XLSRCNN
from transformers import WhisperProcessor, WhisperModel, Wav2Vec2FeatureExtractor, Wav2Vec2Model
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from thesis import MiOModel, train_model, evaluate_model  # Replace with actual import of MiO model
import warnings

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and freeze the MiO model
def load_frozen_mio_model(model_path):
    if not os.path.exists(model_path):
        train_model()
    mio_model = MiOModel().to(device)
    mio_model.load_state_dict(torch.load(model_path))
    print(f"Model loaded from {model_path}")
    # evaluate_model(mio_model)
    mio_model.eval()  # Set to eval mode
    for param in mio_model.parameters():
        param.requires_grad = False  # Freeze all parameters
    return mio_model

# Load Whisper model and processor
whisper_model = WhisperModel.from_pretrained('openai/whisper-base').to(device)
whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-base')

# Load XLS-R model and processor
xlsr_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-1b').to(device)
xlsr_processor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-1b')

# Function to extract CNN features for a single MLAAD audio file
import torchaudio
import torch
from transformers import WhisperProcessor, WhisperModel, Wav2Vec2Processor, Wav2Vec2Model

def preprocess_audio_with_whisper_xlsr(audio_path):
    # Load audio waveform
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)

    # Define LFCC transformation
    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=16000,
        n_lfcc=128,
        speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 400}
    ).to(device)

    # Calculate LFCCs with delta and double-delta
    lfcc = lfcc_transform(waveform).squeeze(0)  # [128, Time]
    delta = torchaudio.functional.compute_deltas(lfcc)  # Delta LFCCs
    double_delta = torchaudio.functional.compute_deltas(delta)  # Double-delta LFCCs
    lfcc_combined = torch.cat((lfcc, delta, double_delta), dim=0)  # Concatenate to [384, Time]

    # Pad or truncate to a consistent time length
    target_length = 3000
    if lfcc_combined.size(1) < target_length:
        padding = target_length - lfcc_combined.size(1)
        lfcc_combined = torch.nn.functional.pad(lfcc_combined, (0, padding), mode='constant', value=0)
    else:
        lfcc_combined = lfcc_combined[:, :target_length]
    
    # Add batch dimension and permute to [batch_size, input_channels, sequence_length]
    lfcc_combined = lfcc_combined.unsqueeze(0).to(device)  # Shape: [1, 384, target_length]
    # lfcc_combined = lfcc_combined.permute(0,2,1)
    # Check and print shapes before passing to CNN extractor
    # print(f"LFCC combined shape after padding: {lfcc_combined.shape}")  # Should be [1, 384, target_length]

    # Whisper feature extraction
    whisper_cnn_extractor = WhisperCNN(input_dim=384, output_dim=120).to(device)
    cnn_features_whisper = whisper_cnn_extractor(lfcc_combined)  # Shape: [1, 120, target_length]

    # XLS-R feature extraction
    xlsr_cnn_extractor = XLSRCNN(input_dim=384, output_dim=120).to(device)
    cnn_features_xlsr = xlsr_cnn_extractor(lfcc_combined)  # Shape: [1, 120, target_length]

    return cnn_features_whisper, cnn_features_xlsr


# Feature extraction function using the frozen MiO model
def extract_mio_features(audio_path, mio_model):
    cnn_features_whisper, cnn_features_xlsr = preprocess_audio_with_whisper_xlsr(audio_path)
    
    # Check that the shapes are correct before passing to the MiO model
    assert cnn_features_whisper.shape[1] == 120, f"Unexpected whisper feature shape: {cnn_features_whisper.shape}"
    assert cnn_features_xlsr.shape[1] == 120, f"Unexpected XLS-R feature shape: {cnn_features_xlsr.shape}"
    
    # Extract MiO features
    with torch.no_grad():
        feature_tensor = mio_model(cnn_features_whisper, cnn_features_xlsr, return_embedding=True)
    
    # Check the final output shape to ensure consistency
    assert feature_tensor.shape[1] == 120, f"Unexpected feature tensor shape: {feature_tensor.shape}"
    return feature_tensor.cpu()

# Load all simpler_meta.csv files into a single DataFrame
def load_combined_meta(root_dir):
    combined_meta = pd.DataFrame()
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == "simpler_meta.csv":
                meta_path = os.path.join(subdir, file)
                meta_df = pd.read_csv(meta_path)
                combined_meta = pd.concat([combined_meta, meta_df], ignore_index=True)
    return combined_meta

# Dataset class for MLAAD, caching features if not already cached
class MLAADDataset(Dataset):
    def __init__(self, meta_df, root_dir, mio_model, label_type="acoustic", cache_dir="mlaad_cached_features", label_mapping=None):
        self.data = meta_df
        self.root_dir = root_dir
        self.mio_model = mio_model
        self.label_type = label_type
        self.cache_dir = cache_dir
        self.label_mapping = label_mapping
        os.makedirs(cache_dir, exist_ok=True)
        print(f"Initialized MLAADDataset with {len(self.data)} samples.")
        print(f"Label mapping: {self.label_mapping}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.root_dir, row['file_path']).replace("./fake/", "")
        
        # Map the label string to an integer
        label = self.label_mapping[row[self.label_type]]
        
        # Cache file path
        cache_file = os.path.join(self.cache_dir, f"{row['file_path'].replace('/', '_')}.pt")

        # Print status of loading/caching
        if os.path.exists(cache_file):
            # print(f"[{idx}] Loading cached features for: {audio_path}")
            feature_tensor = torch.load(cache_file)
        else:
            # print(f"[{idx}] Extracting features for: {audio_path}")
            feature_tensor = extract_mio_features(audio_path, self.mio_model)
            torch.save(feature_tensor, cache_file)
            # print(f"[{idx}] Cached features saved for: {audio_path}")
        
        return feature_tensor, torch.tensor(label, dtype=torch.long)


# Training function for classification heads
def train_classification_head(classifier, train_loader, num_epochs=30, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = classifier(features).squeeze(1)
            # print(f"Outputs shape: {outputs.shape}, Labels shape: {labels.shape}")
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

    print("Training completed.")
    return classifier

def evaluate_classification_head(classifier, eval_loader):
    classifier.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for features, labels in eval_loader:
            features, labels = features.to(device), labels.to(device)
            
            # Forward pass
            outputs = classifier(features)
            if outputs.dim() > 2:
                # Average over the time dimension to get a single prediction per sample
                outputs = outputs.mean(dim=1)  # Adjust as needed to collapse extra dimensions

            preds = torch.argmax(outputs, dim=1)  # Get the index of the max logit for each sample
            
            # Append predictions and labels, ensuring they are 1D arrays
            all_preds.extend(preds.cpu().numpy())  # Already 1D after argmax
            all_labels.extend(labels.cpu().numpy())
    
    # Convert lists to numpy arrays and ensure they are flat
    all_preds = np.array(all_preds).reshape(-1)
    all_labels = np.array(all_labels).reshape(-1)
    
    # Log shapes for debugging
    print(f"Shapes - Predictions: {all_preds.shape}, Labels: {all_labels.shape}")
    
    # Check that both have the same length
    if all_preds.shape != all_labels.shape:
        raise ValueError(f"Shape mismatch: Predictions shape {all_preds.shape}, Labels shape {all_labels.shape}")
    
    # Evaluate using sklearn metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    return accuracy, precision, recall, f1

# Main execution script
if __name__ == "__main__":
    # Load all meta.csv files from MLAAD and combine them
    root_dir = "Dataset/mlaad/MLAADv3/fake"  # Update this to MLAAD dataset path
    combined_meta = load_combined_meta(root_dir)
    
    # Load frozen MiO model for feature extraction
    mio_model_path = "mio_mode.pth"  # Update this with your MiO model path
    frozen_mio_model = load_frozen_mio_model(mio_model_path)
    
    # Separate data for acoustic and vocoder tasks
    acoustic_train_df, acoustic_eval_df = train_test_split(combined_meta, test_size=0.2, stratify=combined_meta["acoustic"])
    vocoder_train_df, vocoder_eval_df = train_test_split(combined_meta, test_size=0.2, stratify=combined_meta["vocoder"])
    
    all_labels = sorted(set(acoustic_train_df["acoustic"].unique()).union(set(acoustic_eval_df["acoustic"].unique())))
    acoustic_label_mapping = {label: idx for idx, label in enumerate(all_labels)}

    all_labels = sorted(set(vocoder_train_df["vocoder"].unique()).union(set(vocoder_eval_df["vocoder"].unique())))
    vocoder_label_mapping = {label: idx for idx, label in enumerate(all_labels)}

    # Define and prepare DataLoaders for both tasks
    train_loader_acoustic = DataLoader(MLAADDataset(acoustic_train_df, root_dir, frozen_mio_model, label_type="acoustic",label_mapping=acoustic_label_mapping), batch_size=16, shuffle=True)
    eval_loader_acoustic = DataLoader(MLAADDataset(acoustic_eval_df, root_dir, frozen_mio_model, label_type="acoustic",label_mapping=acoustic_label_mapping), batch_size=16, shuffle=False)
    
    train_loader_vocoder = DataLoader(MLAADDataset(vocoder_train_df, root_dir, frozen_mio_model, label_type="vocoder",label_mapping=vocoder_label_mapping), batch_size=16, shuffle=True)
    eval_loader_vocoder = DataLoader(MLAADDataset(vocoder_eval_df, root_dir, frozen_mio_model, label_type="vocoder",label_mapping=vocoder_label_mapping), batch_size=16, shuffle=False)

    # Instantiate classification heads
    num_acoustic_classes = combined_meta["acoustic"].nunique()
    num_vocoder_classes = combined_meta["vocoder"].nunique()
    
    acoustic_classifier = nn.Sequential(
        nn.Linear(120, 64),  # Adjust input size as necessary
        nn.ReLU(),
        nn.Linear(64, num_acoustic_classes)
    ).to(device)
    
    vocoder_classifier = nn.Sequential(
        nn.Linear(120, 64),  # Adjust input size as necessary
        nn.ReLU(),
        nn.Linear(64, num_vocoder_classes)
    ).to(device)

    # Train acoustic classifier
    print("Training Acoustic Classifier")
    train_classification_head(acoustic_classifier, train_loader_acoustic)

    # Train vocoder classifier
    print("Training Vocoder Classifier")
    train_classification_head(vocoder_classifier, train_loader_vocoder)

    print("Evaluating Acoustic Classifier")
    evaluate_classification_head(acoustic_classifier, eval_loader_acoustic)

    # Train vocoder classifier
    print("Evaluating Vocoder Classifier")
    evaluate_classification_head(vocoder_classifier, eval_loader_vocoder)
