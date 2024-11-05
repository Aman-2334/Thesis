import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2Processor, WhisperProcessor
from transformers import Wav2Vec2Model, WhisperModel

# Paths for audio and metadata
audio_dir = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_train\\flac'  # Adjust according to your dataset structure
metadata_path = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt'  # CSV or TXT with file paths and labels

# Load processors
xlsr_processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-xls-r-1b')
whisper_processor = WhisperProcessor.from_pretrained('openai/whisper-base')

# Load Model
whisper_model = WhisperModel.from_pretrained('openai/whisper-base')
xlsr_model = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-1b')

# Custom Dataset class for ASVspoof 2019
class ASVspoofDataset(Dataset):
    def __init__(self, audio_dir, metadata_path, processor, sampling_rate=16000):
        self.audio_dir = audio_dir
        self.metadata = self.load_metadata(metadata_path)
        self.processor = processor
        self.sampling_rate = sampling_rate

    def load_metadata(self, metadata_path):
        # Assumes metadata file has two columns: 'filename', 'label'
        with open(metadata_path, 'r') as f:
            data = [line.strip().split() for line in f.readlines()]
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        filename, label = self.metadata[idx]
        audio_path = os.path.join(self.audio_dir, filename + '.flac')
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.sampling_rate)
            waveform = resampler(waveform)

        # Process audio to get input tensors
        inputs = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)
        return inputs.input_values.squeeze(0), torch.tensor(int(label))  # Adjust label parsing as needed

# Create DataLoader
train_dataset = ASVspoofDataset(audio_dir, metadata_path, xlsr_processor)  # Use one processor; adjust as needed
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Function to extract features from PTMs
def extract_features(model, input_values):
    with torch.no_grad():
        outputs = model(input_values).last_hidden_state
        return torch.mean(outputs, dim=1)  # Mean pooling

# Training loop adjusted for ASVspoof 2019 dataset
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Extract features from Whisper and XLS-R
            whisper_features = extract_features(whisper_model, inputs)
            xlsr_features = extract_features(xlsr_model, inputs)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(whisper_features, xlsr_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim=120):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # Assuming input shape [batch_size, seq_length, input_dim]
        x = x.permute(0, 2, 1)  # Change shape to [batch_size, input_dim, seq_length] for Conv1d
        x = self.cnn(x)
        x = torch.mean(x, dim=2)  # Mean pooling to flatten the representation
        return x

# Example of using this in the MiO model
class MiOModel(nn.Module):
    def __init__(self, whisper_input_dim=512, xlsr_input_dim=1280, cnn_output_dim=120, output_dim=2):
        super(MiOModel, self).__init__()
        self.whisper_cnn = CNNFeatureExtractor(whisper_input_dim, cnn_output_dim)
        self.xlsr_cnn = CNNFeatureExtractor(xlsr_input_dim, cnn_output_dim)
        self.bilinear_pooling = nn.Bilinear(cnn_output_dim, cnn_output_dim, cnn_output_dim)
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=1)
        )

    def forward(self, whisper_features, xlsr_features):
        whisper_cnn_output = self.whisper_cnn(whisper_features)
        xlsr_cnn_output = self.xlsr_cnn(xlsr_features)
        bp_result = self.bilinear_pooling(whisper_cnn_output, xlsr_cnn_output)
        output = self.fc(bp_result)
        return output

# Integrating into the training loop
model = MiOModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_model(model, train_loader, criterion, optimizer)