import os
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

# Load XLS-R model and processor
print("Loading XLS-R model and processor...")
xlsr_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-1b')
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
        print(f"Loaded audio file: {audio_path}")

        # Resample if needed
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)
            print(f"Resampled audio to {self.sampling_rate} Hz.")

        # Remove extra dimensions if needed
        waveform = waveform.squeeze(0)

        # Process the waveform using XLS-R processor to obtain input values
        inputs = xlsr_feature_extractor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        
        return inputs.input_values.squeeze(0), label  # Remove batch dimension for single sample

def collate_fn(batch):
    inputs, labels = zip(*batch)  # Separate inputs and labels
    # Find the maximum length in the batch
    max_length = max(input.size(0) for input in inputs)
    # Pad all inputs to the maximum length
    inputs_padded = [F.pad(input, (0, max_length - input.size(0))) for input in inputs]
    inputs_padded = torch.stack(inputs_padded)  # Stack into a single tensor
    labels = torch.tensor(labels)  # Convert labels to tensor
    return inputs_padded, labels

# Define paths
audio_dir = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_train\\flac'
# CSV or TXT with file paths and labels
metadata_path = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt'

# Create the dataset and DataLoader
dataset = ASVspoofXLSRDataset(audio_dir, metadata_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Extract features and inspect
print("Starting XLS-R feature extraction...")
for batch_idx, (input_values, labels) in enumerate(dataloader):
    print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")

    with torch.no_grad():
        # Pass waveforms through XLS-R model to extract encoder features
        xlsr_outputs = xlsr_model(input_values)
        
        # Extract last hidden state from XLS-R encoder
        xlsr_features = xlsr_outputs.last_hidden_state

    # Print shape of extracted features and labels to confirm
    print("XLS-R encoder feature shape:", xlsr_features.shape)  # [batch_size, seq_length, hidden_size]
    print("Labels:", labels)
    
    # For demonstration, process just one batch
    break

print("XLS-R feature extraction completed.")
