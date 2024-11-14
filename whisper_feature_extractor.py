import os
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import pad
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
import os
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F

# Define LFCC parameters
class ASVspoofWhisperDataset(Dataset):
    def __init__(self, audio_dir, protocol_path, sampling_rate=16000, target_length=3000):
        self.audio_dir = audio_dir
        self.metadata = self.load_metadata(protocol_path)
        self.sampling_rate = sampling_rate
        self.target_length = target_length
        self.lfcc_transform = torchaudio.transforms.LFCC(
            sample_rate=sampling_rate, 
            n_lfcc=128,  # 128-dimensional LFCCs
            speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 400}
        )
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

        # Resample if needed
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # Remove extra dimensions if needed
        waveform = waveform.squeeze(0)

        # Extract LFCC features
        lfcc = self.lfcc_transform(waveform).squeeze(0)  # [128, Time]
        delta = torchaudio.functional.compute_deltas(lfcc)  # Delta LFCCs
        double_delta = torchaudio.functional.compute_deltas(delta)  # Double-delta LFCCs

        # Concatenate LFCCs, delta, and double-delta features
        lfcc_combined = torch.cat((lfcc, delta, double_delta), dim=0)  # [384, Time]

        # Pad or truncate to target length
        if lfcc_combined.size(1) < self.target_length:
            padding = self.target_length - lfcc_combined.size(1)
            lfcc_combined = F.pad(lfcc_combined, (0, padding), mode='constant', value=0)
        else:
            lfcc_combined = lfcc_combined[:, :self.target_length]

        return lfcc_combined, label


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
        # print(f"Input to CNN extractor before permute: {x.shape}")
        x = x.permute(0, 1, 2)  # Change shape to [batch_size, input_dim, seq_length]
        # print(f"Input to CNN extractor after permute: {x.shape}")
        # Apply the 1D CNN and MaxPool
        x = self.cnn(x)
        
        # Optional: Use mean pooling to get a fixed-size output
        x = torch.mean(x, dim=2)  # [batch_size, output_dim]
        
        return x
    
# Extract features and inspect
def whisper_batch_generator(dataloader, cache_dir='cache_whisper_batches', sampling_rate=16000, target_length=3000):
    print("Starting LFCC feature extraction and caching to disk...")
    os.makedirs(cache_dir, exist_ok=True)  # Create a directory for cached batches if it doesn't exist
    
    # LFCC transform setup
    lfcc_transform = torchaudio.transforms.LFCC(
        sample_rate=sampling_rate, 
        n_lfcc=128,  # 128-dimensional LFCCs
        speckwargs={"n_fft": 512, "hop_length": 160, "win_length": 400}
    ).to(device)
    cnn_extractor = CNNFeatureExtractor(input_dim=384, output_dim=120).to(device)  # Adjust input_dim to 384 for LFCC + delta + double-delta

    for batch_idx, (waveforms, labels) in enumerate(dataloader):
        print(f"Processing and caching batch {batch_idx + 1}/{len(dataloader)}")

        # Move data to GPU if available
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        # LFCC feature extraction for each waveform in the batch
        lfcc_features_batch = []
        for waveform in waveforms:
            lfcc = lfcc_transform(waveform).squeeze(0)  # Extract LFCCs [128, Time]
            delta = torchaudio.functional.compute_deltas(lfcc)  # Delta LFCCs
            double_delta = torchaudio.functional.compute_deltas(delta)  # Double-delta LFCCs
            lfcc_combined = torch.cat((lfcc, delta, double_delta), dim=0)  # Concatenate [384, Time]
            # print(f"LFCC combined shape (after concat): {lfcc_combined.shape}")

            # Reshape or slice to match target dimensions
            if lfcc_combined.dim() > 2:
                lfcc_combined = lfcc_combined.view(384, -1)  # Flatten any extra dimensions

            # Pad or truncate to target length
            if lfcc_combined.size(1) < target_length:
                padding = target_length - lfcc_combined.size(1)
                lfcc_combined = pad(lfcc_combined, (0, padding), mode='constant', value=0)
            else:
                lfcc_combined = lfcc_combined[:, :target_length]
            
            lfcc_features_batch.append(lfcc_combined)
        
        # Stack features to create a batch tensor
        lfcc_features_batch = torch.stack(lfcc_features_batch).to(device)
        # print(f"LFCC features batch shape before passing to CNN extractor: {lfcc_features_batch.shape}")

        # Ensure lfcc_features_batch has exactly 3 dimensions for CNN extractor
        if lfcc_features_batch.dim() == 4:
            lfcc_features_batch = lfcc_features_batch.squeeze(1)

        # Forward pass through CNN extractor
        cnn_features = cnn_extractor(lfcc_features_batch)

        # Move features and labels to CPU before saving
        cnn_features = cnn_features.cpu()
        labels = labels.cpu()

        # Save the features and labels to disk
        batch_cache_path = os.path.join(cache_dir, f'batch_{batch_idx}.pt')
        torch.save({'features': cnn_features, 'labels': labels}, batch_cache_path)

        # Clear GPU memory after each batch
        del waveforms, lfcc_features_batch, cnn_features, labels
        torch.cuda.empty_cache()

    print("LFCC feature extraction and caching to disk completed.")