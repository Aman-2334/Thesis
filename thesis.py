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

def train_model_single_batch():
    batch_size = 32
    dataset_whisper = ASVspoofWhisperDataset(train_audio_dir, train_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(train_audio_dir, train_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Get the first batch from both Whisper and XLS-R
    whisper_batch = next(iter(whisper_batch_generator(dataloader_whisper)))
    xlsr_batch = next(iter(xlsr_batch_generator(dataloader_xlsr)))
    cnn_features_whisper, labels_whisper = whisper_batch
    cnn_features_xlsr, labels_xlsr = xlsr_batch

    assert torch.equal(labels_whisper, labels_xlsr), "Mismatch in labels for the single batch"

    cnn_features_whisper = cnn_features_whisper.to(device)
    cnn_features_xlsr = cnn_features_xlsr.to(device)
    labels_whisper = labels_whisper.to(device)
    labels_xlsr = labels_xlsr.to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass through MiO model
        outputs = mio_model(cnn_features_whisper, cnn_features_xlsr)

        # Compute loss
        loss = criterion(outputs, labels_whisper)

        # Backward pass and optimization
        loss.backward(retain_graph=True)
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training on a single batch completed.")
    # Save the model after training
    torch.save(mio_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

def extract_features(dataloader_whisper, dataloader_xlsr):
    # Initialize lists to store features and labels
    whisper_features_list = []
    xlsr_features_list = []
    labels_list = []

    # Extract features from Whisper and XLS-R once
    for (cnn_features_whisper, labels_whisper), (cnn_features_xlsr, labels_xlsr) in zip(
        whisper_batch_generator(dataloader_whisper), xlsr_batch_generator(dataloader_xlsr)
    ):
        # Ensure labels match
        assert torch.equal(labels_whisper, labels_xlsr), "Mismatch in labels between Whisper and XLS-R batches"
        
        # Store features and labels
        whisper_features_list.append(cnn_features_whisper)
        xlsr_features_list.append(cnn_features_xlsr)
        labels_list.append(labels_whisper)

    print("Feature extraction completed.")
    return whisper_features_list, xlsr_features_list, labels_list

def train_model():
    # Batch size and data loaders
    batch_size = 32
    dataset_whisper = ASVspoofWhisperDataset(train_audio_dir, train_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(train_audio_dir, train_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Extract and store features once
    whisper_features_list, xlsr_features_list, labels_list = extract_features(dataloader_whisper, dataloader_xlsr)

    for epoch in range(num_epochs):
        running_loss = 0.0

        # Loop through stored features and labels for each epoch
        for batch_idx, (cnn_features_whisper, cnn_features_xlsr, labels) in enumerate(
            zip(whisper_features_list, xlsr_features_list, labels_list)
        ):
            cnn_features_whisper = cnn_features_whisper.to(device)
            cnn_features_xlsr = cnn_features_xlsr.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass through MiO model
            outputs = mio_model(cnn_features_whisper, cnn_features_xlsr)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward(retain_graph=True)
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
    # train_model_single_batch()
    # pass
#=========================================================================TESTING=========================================================================
# Define paths
test_audio_dir = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_eval\\flac'
# CSV or TXT with file paths and labels
test_metadata_path = 'dataset\\ASVspoof2019\\LA\\ASVspoof2019_LA_cm_protocols\\ASVspoof2019.LA.cm.eval.trl.txt'

def evaluate_model():
    batch_size = 32
    dataset_whisper = ASVspoofWhisperDataset(test_audio_dir, test_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(test_audio_dir, test_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Step 1: Extract features once for evaluation
    whisper_features_list, xlsr_features_list, labels_list = extract_features(dataloader_whisper, dataloader_xlsr)

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

def test_model_single_batch():
    mio_model.eval()  # Set model to evaluation mode

    batch_size = 32
    dataset_whisper = ASVspoofWhisperDataset(test_audio_dir, test_metadata_path)
    dataloader_whisper = DataLoader(dataset_whisper, batch_size=batch_size, shuffle=False)
    dataset_xlsr = ASVspoofXLSRDataset(test_audio_dir, test_metadata_path)
    dataloader_xlsr = DataLoader(dataset_xlsr, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    whisper_batch = next(iter(whisper_batch_generator(dataloader_whisper)))
    xlsr_batch = next(iter(xlsr_batch_generator(dataloader_xlsr)))
    cnn_features_whisper, labels_whisper = whisper_batch
    cnn_features_xlsr, labels_xlsr = xlsr_batch

    cnn_features_whisper = cnn_features_whisper.to(device)
    cnn_features_xlsr = cnn_features_xlsr.to(device)
    labels_whisper = labels_whisper.to(device)
    labels_xlsr = labels_xlsr.to(device)

    with torch.no_grad():
        # Forward pass through MiO model
        outputs = mio_model(cnn_features_whisper, cnn_features_xlsr)
        _, predicted = torch.max(outputs, 1)

        # Move predictions and labels to CPU for metrics calculation
        all_preds = predicted.cpu().numpy()
        all_labels = labels_whisper.cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')

    print(f"Single Batch Test Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

evaluate_model()
# test_model_single_batch()
# print("torch cuda",torch.cuda.is_available())

#=========================================================================TWO STAGE PROCESS=========================================================================
for param in mio_model.parameters():
    param.requires_grad = False

print("MiO model backbone has been frozen for feature extraction.")

class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# Define classification heads for different tasks
input_dim = 120  # Set to the output dimension of MiO's CNN output
input_type_head = ClassificationHead(input_dim, num_classes=3)  # e.g., speech, text, bonafide
acoustic_model_head = ClassificationHead(input_dim, num_classes=5)  # Number of acoustic model classes
vocoder_head = ClassificationHead(input_dim, num_classes=5)  # Number of vocoder classes

# Move classification heads to the same device as MiO model
input_type_head.to(device)
acoustic_model_head.to(device)
vocoder_head.to(device)

def train_classification_head(head, dataloader, criterion, optimizer, num_epochs=10):
    head.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (embeddings, labels) in enumerate(dataloader):
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = head(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}')

# Optimizers and Loss for each head
criterion = nn.CrossEntropyLoss()

# Optimizers for each task-specific head
input_type_optimizer = torch.optim.Adam(input_type_head.parameters(), lr=0.001)
acoustic_model_optimizer = torch.optim.Adam(acoustic_model_head.parameters(), lr=0.001)
vocoder_optimizer = torch.optim.Adam(vocoder_head.parameters(), lr=0.001)

# Assuming dataloaders for each task
train_classification_head(input_type_head, input_type_dataloader, criterion, input_type_optimizer)
train_classification_head(acoustic_model_head, acoustic_model_dataloader, criterion, acoustic_model_optimizer)
train_classification_head(vocoder_head, vocoder_dataloader, criterion, vocoder_optimizer)
