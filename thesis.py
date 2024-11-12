import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from whisper_feature_extractor import whisper_batch_generator, ASVspoofWhisperDataset
from xlsr_feature_extractor import xlsr_batch_generator, ASVspoofXLSRDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# torch.backends.cudnn.benchmark = True

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
#=========================================================================TRAINING=========================================================================
# Define paths
train_audio_dir = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_train\\flac'
# CSV or TXT with file paths and labels
train_metadata_path = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.train.trn.txt'
model_save_path = "mio_model.pth"

# Initialize the MiO model
cnn_output_dim = 120  # Output dimension from the CNN feature extractor
output_dim = 2  # Number of classes for classification
mio_model = MiOModel(cnn_output_dim=cnn_output_dim, output_dim=output_dim).to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(mio_model.parameters(), lr=0.001)

num_epochs = 10  # Adjust based on your requirements

import torch
import os

def extract_features(dataloader_whisper, dataloader_xlsr, whisper_cache_dir, xlsr_cache_dir):
    # Initialize lists to store features and labels on CPU
    whisper_features_list = []
    xlsr_features_list = []
    labels_list = []

    # Ensure the cached directories exist
    if not os.path.exists(whisper_cache_dir):
        print(f"Whisper cache directory '{whisper_cache_dir}' does not exist.")
        whisper_batch_generator(dataloader_whisper,whisper_cache_dir)
    if not os.path.exists(xlsr_cache_dir):
        print(f"XLS-R cache directory '{xlsr_cache_dir}' does not exist.")
        xlsr_batch_generator(dataloader_xlsr,xlsr_cache_dir)

    # Get sorted list of batch files in each cache directory
    whisper_batch_files = sorted(os.listdir(whisper_cache_dir))
    xlsr_batch_files = sorted(os.listdir(xlsr_cache_dir))

    # Ensure the number of batches matches between the two caches
    assert len(whisper_batch_files) == len(xlsr_batch_files), "Mismatch in the number of Whisper and XLS-R cached batches."

    # Load each batch from the cached directories
    for whisper_file, xlsr_file in zip(whisper_batch_files, xlsr_batch_files):
        # Load the cached Whisper and XLS-R batches
        whisper_batch_path = os.path.join(whisper_cache_dir, whisper_file)
        xlsr_batch_path = os.path.join(xlsr_cache_dir, xlsr_file)

        whisper_data = torch.load(whisper_batch_path)
        xlsr_data = torch.load(xlsr_batch_path)

        # Ensure labels match
        assert torch.equal(whisper_data['labels'], xlsr_data['labels']), "Mismatch in labels between Whisper and XLS-R batches"

        # Enable gradient tracking
        whisper_features = whisper_data['features'].requires_grad_(True)
        xlsr_features = xlsr_data['features'].requires_grad_(True)

        whisper_features_list.append(whisper_features)
        xlsr_features_list.append(xlsr_features)
        labels_list.append(whisper_data['labels'])

    return whisper_features_list, xlsr_features_list, labels_list

def train_model():
    # Batch size and data loaders
    batch_size=32
    dataset_whisper = ASVspoofWhisperDataset(train_audio_dir, train_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(train_audio_dir, train_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    whisper_cache_dir='train_cache_whisper_batches'
    xlsr_cache_dir='train_cache_xlsr_batches'
    # Extract and store features once
    whisper_features_list, xlsr_features_list, labels_list = extract_features(dataloader_whisper, dataloader_xlsr, whisper_cache_dir, xlsr_cache_dir)

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Loop through stored features and labels for each epoch
        for batch_idx, (cnn_features_whisper, cnn_features_xlsr, labels) in enumerate(
            zip(whisper_features_list, xlsr_features_list, labels_list)
        ):
            cnn_features_whisper = cnn_features_whisper.to(device).requires_grad_(True)  # Enable gradients
            cnn_features_xlsr = cnn_features_xlsr.to(device).requires_grad_(True)  # Enable gradients
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass through MiO model
            outputs = mio_model(cnn_features_whisper, cnn_features_xlsr)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print statistics every 50 batches as an example
            if batch_idx % 50 == 49:  # Adjust the logging frequency as needed
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}], Loss: {running_loss / 50:.4f}')
                running_loss = 0.0

    print("Training completed.")
    torch.save(mio_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


if os.path.exists(model_save_path):
    # Load the model if it exists
    mio_model.load_state_dict(torch.load(model_save_path))
    print(f"Model loaded from {model_save_path}")
else:
    # Train the model if it doesn't exist
    train_model()

#=========================================================================TESTING=========================================================================
# Define paths
test_audio_dir = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_eval\\flac'
# CSV or TXT with file paths and labels
test_metadata_path = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.eval.trl.txt'

def evaluate_model():
    batch_size = 16
    dataset_whisper = ASVspoofWhisperDataset(test_audio_dir, test_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(test_audio_dir, test_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    whisper_cache_dir='eval_cache_whisper_batches'
    xlsr_cache_dir='eval_cache_xlsr_batches'
    # Extract and store features once
    whisper_features_list, xlsr_features_list, labels_list = extract_features(dataloader_whisper, dataloader_xlsr, whisper_cache_dir, xlsr_cache_dir)

    mio_model.eval()  # Set the model to evaluation mode
    all_preds = []
    all_labels = []

    with torch.no_grad():  # Disable gradient computation for evaluation
        # Iterate over stored features and labels
        for cnn_features_whisper, cnn_features_xlsr, labels in zip(
            whisper_features_list, xlsr_features_list, labels_list
        ):
            cnn_features_whisper = cnn_features_whisper.to(device)
            cnn_features_xlsr = cnn_features_xlsr.to(device)
            labels = labels.to(device)

            # Forward pass through the MiO model
            outputs = mio_model(cnn_features_whisper, cnn_features_xlsr)
            _, predicted = torch.max(outputs, 1)  # Get predicted class

            # Append predictions and labels to lists
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='binary', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='binary', zero_division=1)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return accuracy, precision, recall, f1

evaluate_model()
# print("torch cuda",torch.cuda.is_available())