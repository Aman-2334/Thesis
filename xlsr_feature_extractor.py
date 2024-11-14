import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load XLS-R model and processor
print("Loading XLS-R model and processor...")
xlsr_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-1b').to(device)
xlsr_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-1b')
print("XLS-R model and processor loaded successfully.")

# ASVspoof2019 Dataset class for extracting XLS-R features
class ASVspoofXLSRDataset(Dataset):
    def __init__(self, audio_dir, protocol_path, sampling_rate=16000):
        self.audio_dir = audio_dir
        self.metadata = self.load_metadata(protocol_path)
        self.sampling_rate = sampling_rate
        print(f"Dataset initialized with {len(self.metadata)} samples.")

    def load_metadata(self, protocol_path):
        data = []
        with open(protocol_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                filename = parts[1]  # Second column is the file name
                label = 0 if parts[4] == 'bonafide' else 1  # 'bonafide' is 0 (real), 'spoof' is 1
                data.append((filename, label))
        print("Metadata loaded successfully.")
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename, label = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, filename + '.flac')
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(audio_path)
        # print(f"Loaded audio file: {audio_path}")

        # Resample if needed
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)
            # print(f"Resampled audio to {self.sampling_rate} Hz.")

        # Remove extra dimensions if needed
        waveform = waveform.squeeze(0)

        # Process the waveform using XLS-R processor to obtain input values
        inputs = xlsr_feature_extractor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        
        return inputs.input_values.squeeze(0), label  # Remove batch dimension for single sample

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim=120):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Input shape: [batch_size, seq_length, input_dim]
        
        # Apply the 1D CNN and MaxPool
        x = self.cnn(x)
        
        # Optional: Use mean pooling to get a fixed-size output
        x = torch.mean(x, dim=2)  # [batch_size, output_dim]
        
        return x

# Extract features and inspect
def xlsr_batch_generator(dataloader, cache_dir='cache_xlsr_batches'):
    print("Starting XLS-R feature extraction and caching to disk...")
    os.makedirs(cache_dir, exist_ok=True)  # Create cache directory if it doesn't exist
    cnn_extractor = CNNFeatureExtractor(input_dim=1280, output_dim=120).to(device)

    for batch_idx, (input_values, labels) in enumerate(dataloader):
        print(f"Processing and caching XLS-R batch {batch_idx + 1}/{len(dataloader)}")
        # batch_path = os.path.join(cache_dir,f'batch_{batch_idx}.pt')
        # if(os.path.exists(batch_path)):
        #     continue
        # Move data to GPU
        input_values = input_values.to(device)
        labels = labels.to(device)

        # Forward pass through XLS-R and CNN extractor
        xlsr_outputs = xlsr_model(input_values)
        xlsr_features = xlsr_outputs.last_hidden_state
        xlsr_features = xlsr_features.permute(0, 2, 1) 
        cnn_features = cnn_extractor(xlsr_features)

        # Move features and labels to CPU before saving
        cnn_features = cnn_features.cpu()
        labels = labels.cpu()

        # Save features and labels to disk
        batch_cache_path = os.path.join(cache_dir, f'batch_{batch_idx}.pt')
        torch.save({'features': cnn_features, 'labels': labels}, batch_cache_path)

        # Clear GPU memory after each batch
        del input_values, xlsr_features, xlsr_outputs, cnn_features, labels
        torch.cuda.empty_cache()
    print("XLS-R feature extraction and caching to disk completed.")