import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from whisper_feature_extractor import whisper_batch_generator
from xlsr_feature_extractor import xlsr_batch_generator

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

# Initialize the MiO model
cnn_output_dim = 120  # Output dimension from the CNN feature extractor
output_dim = 2  # Number of classes for classification
mio_model = MiOModel(cnn_output_dim=cnn_output_dim, output_dim=output_dim)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for multi-class classification
optimizer = optim.Adam(mio_model.parameters(), lr=0.001)

num_epochs = 10  # Adjust based on your requirements

def train_model():
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Loop over batches returned from both feature extractors
        for batch_idx, ((cnn_features_whisper, labels_whisper), (cnn_features_xlsr, labels_xlsr)) in enumerate(
            zip(whisper_batch_generator(), xlsr_batch_generator())
        ):
            # Ensure labels match between whisper and xlsr features for each batch
            assert torch.equal(labels_whisper, labels_xlsr), f"Mismatch in labels at batch {batch_idx}"

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