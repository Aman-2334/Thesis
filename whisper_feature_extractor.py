import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import WhisperModel, WhisperProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Whisper model and processor
print("Loading Whisper model and processor...")
whisper_model = WhisperModel.from_pretrained('openai/whisper-base').to(device)
whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-base')
print("Whisper model and processor loaded successfully.")

# ASVspoof2019 Dataset class for extracting Whisper features
class ASVspoofWhisperDataset(Dataset):
    def __init__(self, audio_dir, protocol_path, sampling_rate=16000, target_length=3000):
        self.audio_dir = audio_dir
        self.metadata = self.load_metadata(protocol_path)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
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
        print(f"Loaded audio file: {audio_path}")

        # Resample if needed
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)
            print(f"Resampled audio to {self.sampling_rate} Hz.")

        # Remove extra dimensions if needed
        waveform = waveform.squeeze(0)

        # Convert waveform to mel-spectrogram using Whisper processor
        mel_features = whisper_processor.feature_extractor(
            waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_features.squeeze(0)  # Remove batch dimension

        # Pad or truncate to target length
        if mel_features.size(-1) < self.target_length:
            padding = self.target_length - mel_features.size(-1)
            mel_features = F.pad(mel_features, (0, padding), mode='constant', value=0)
            print(f"Padded mel-spectrogram to length {self.target_length}.")
        else:
            mel_features = mel_features[:, :self.target_length]
            print(f"Truncated mel-spectrogram to length {self.target_length}.")

        return mel_features, label

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
        # Rearrange to [batch_size, input_dim, seq_length] for Conv1d
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, input_dim, seq_length]
        
        # Apply the 1D CNN and MaxPool
        x = self.cnn(x)
        
        # Optional: Use mean pooling to get a fixed-size output
        x = torch.mean(x, dim=2)  # [batch_size, output_dim]
        
        return x
    
# Extract features and inspect
def whisper_batch_generator(dataloader):
    print("Starting feature extraction...")
    for batch_idx, (mel_features, labels) in enumerate(dataloader):
        print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

        mel_features = mel_features.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            # Pass mel-spectrograms through Whisper model
            whisper_encoder_outputs = whisper_model.encoder(mel_features)
            whisper_features = whisper_encoder_outputs.last_hidden_state  # Extract last hidden state

        # Print shape of extracted features and labels to confirm
        print("Whisper encoder feature shape:", whisper_features.shape)  # [batch_size, seq_length, hidden_size]
        print("Labels:", labels)
        # whisper_features is the output from whisper model with shape [batch_size, seq_length, hidden_size]
        cnn_extractor = CNNFeatureExtractor(input_dim=whisper_features.shape[2], output_dim=120).to(device) #input_dim = hidden_size
        cnn_features = cnn_extractor(whisper_features)
        print("CNN-extracted features shape:", cnn_features.shape)
        yield cnn_features,labels
    print("whisper feature extraction with 1D CNN completed.")

