import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from whisper_feature_extractor import whisper_batch_generator, ASVspoofWhisperDataset
from xlsr_feature_extractor import xlsr_batch_generator, ASVspoofXLSRDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class MiOModel(nn.Module):
    def __init__(self, cnn_output_dim=120, fcn_hidden_dim=128, output_dim=2):
        super(MiOModel, self).__init__()
        
        # Bilinear pooling layer
        self.bilinear_pooling = nn.Bilinear(cnn_output_dim, cnn_output_dim, cnn_output_dim)
        
        # Fully connected network after bilinear pooling
        self.fc = nn.Sequential(
            nn.Linear(cnn_output_dim, fcn_hidden_dim),
            nn.ReLU(),
            nn.Linear(fcn_hidden_dim, output_dim)
        )
        
        # Softmax for classification
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, cnn_features_whisper, cnn_features_xlsr):
        # Perform bilinear pooling on the CNN outputs from Whisper and XLS-R
        bilinear_output = self.bilinear_pooling(cnn_features_whisper, cnn_features_xlsr)
        
        # Pass through fully connected layers
        fc_output = self.fc(bilinear_output)
        
        # Apply softmax for final classification probabilities
        output = self.softmax(fc_output)
        
        return output

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
tain_audio_dir = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_train\\flac'
# CSV or TXT with file paths and labels
train_metadata_path = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt'

# Initialize the MiO model
cnn_output_dim = 120  # Output dimension from the CNN feature extractor
output_dim = 2  # Number of classes for classification
mio_model = MiOModel(cnn_output_dim=cnn_output_dim, output_dim=output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(mio_model.parameters(), lr=0.001)

num_epochs = 10  # Adjust based on your requirements

def train_model():
    batch_size = 32
    dataset_whisper = ASVspoofWhisperDataset(tain_audio_dir, train_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(tain_audio_dir, train_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Loop over batches returned from both feature extractors
        for batch_idx, ((cnn_features_whisper, labels_whisper), (cnn_features_xlsr, labels_xlsr)) in enumerate(
            zip(whisper_batch_generator(dataloader_whisper), xlsr_batch_generator(dataloader_xlsr))
        ):
            # Ensure labels match between whisper and xlsr features for each batch
            assert torch.equal(labels_whisper, labels_xlsr), f"Mismatch in labels at batch {batch_idx}"

            cnn_features_whisper = cnn_features_whisper.to(device)
            cnn_features_xlsr = cnn_features_xlsr.to(device)
            labels_whisper = labels_whisper.to(device)
            labels_xlsr = labels_xlsr.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass through MiO model
            outputs = mio_model(cnn_features_whisper, cnn_features_xlsr)

            # Compute loss
            loss = criterion(outputs, labels_whisper)  # or labels_xlsr, both are the same

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Accumulate loss for monitoring
            running_loss += loss.item()

            # Print statistics every 50 batches as an example
            if batch_idx % 50 == 49:  # Adjust the logging frequency as needed
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {running_loss / 50:.4f}')
                running_loss = 0.0

    print("Training completed.")

# print(torch.cuda.is_available())
train_model()