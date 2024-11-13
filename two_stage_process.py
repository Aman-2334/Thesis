import os
import torch
import pandas as pd
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
from thesis import MiOModel  # Replace with actual import of MiO model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and freeze the MiO model
def load_frozen_mio_model(model_path):
    mio_model = MiOModel().to(device)
    mio_model.load_state_dict(torch.load(model_path))
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
def preprocess_audio_with_whisper_xlsr(audio_path):
    # Load audio waveform
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.to(device)

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000).to(device)
        waveform = resampler(waveform)

    # Whisper feature extraction
    mel_features = whisper_processor.feature_extractor(
        waveform.cpu().squeeze(0).numpy(),
        sampling_rate=16000,
        return_tensors="pt"
    ).input_features.to(device)
    
    with torch.no_grad():
        whisper_encoder_outputs = whisper_model.encoder(mel_features)
        whisper_features = whisper_encoder_outputs.last_hidden_state

    # Whisper CNN extraction
    whisper_cnn_extractor = WhisperCNN(input_dim=whisper_features.shape[2], output_dim=120).to(device)
    cnn_features_whisper = whisper_cnn_extractor(whisper_features)

    # XLS-R feature extraction
    xlsr_inputs = xlsr_processor(waveform.cpu().squeeze(0).numpy(), sampling_rate=16000, return_tensors="pt", padding=True)
    xlsr_input_values = xlsr_inputs.input_values.to(device)

    with torch.no_grad():
        xlsr_outputs = xlsr_model(xlsr_input_values)
        xlsr_features = xlsr_outputs.last_hidden_state

    # XLS-R CNN extraction
    xlsr_cnn_extractor = XLSRCNN(input_dim=xlsr_features.shape[2], output_dim=120).to(device)
    cnn_features_xlsr = xlsr_cnn_extractor(xlsr_features)

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
    def __init__(self, meta_df, root_dir, mio_model, label_type="acoustic", cache_dir="mlaad_cached_features"):
        self.data = meta_df
        self.root_dir = root_dir
        self.mio_model = mio_model
        self.label_type = label_type
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        # Create a mapping from label strings to integers
        self.label_mapping = {label: idx for idx, label in enumerate(self.data[self.label_type].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = os.path.join(self.root_dir, row['file_path']).replace("./fake/", "")
        
        # Map the label string to an integer
        label = self.label_mapping[row[self.label_type]]
        
        # Cache file path
        cache_file = os.path.join(self.cache_dir, f"{row['file_path'].replace('/', '_')}.pt")
        
        if os.path.exists(cache_file):
            feature_tensor = torch.load(cache_file)
        else:
            feature_tensor = extract_mio_features(audio_path, self.mio_model)
            torch.save(feature_tensor, cache_file)
        
        return feature_tensor, torch.tensor(label, dtype=torch.long)


# Training function for classification heads
def train_classification_head(classifier, train_loader, num_epochs=10, learning_rate=0.001):
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
            outputs = classifier(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    mio_model_path = "mio_model.pth"  # Update this with your MiO model path
    frozen_mio_model = load_frozen_mio_model(mio_model_path)
    
    # Separate data for acoustic and vocoder tasks
    acoustic_train_df, acoustic_eval_df = train_test_split(combined_meta, test_size=0.2, stratify=combined_meta["acoustic"])
    vocoder_train_df, vocoder_eval_df = train_test_split(combined_meta, test_size=0.2, stratify=combined_meta["vocoder"])

    # Define and prepare DataLoaders for both tasks
    train_loader_acoustic = DataLoader(MLAADDataset(acoustic_train_df, root_dir, frozen_mio_model, label_type="acoustic"), batch_size=16, shuffle=True)
    eval_loader_acoustic = DataLoader(MLAADDataset(acoustic_eval_df, root_dir, frozen_mio_model, label_type="acoustic"), batch_size=16, shuffle=False)
    
    train_loader_vocoder = DataLoader(MLAADDataset(vocoder_train_df, root_dir, frozen_mio_model, label_type="vocoder"), batch_size=16, shuffle=True)
    eval_loader_vocoder = DataLoader(MLAADDataset(vocoder_eval_df, root_dir, frozen_mio_model, label_type="vocoder"), batch_size=16, shuffle=False)

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
