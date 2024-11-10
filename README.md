# Two-Stage Audio Deepfake Detection with MiO Model

This repository implements a two-stage audio deepfake detection process based on the paper *Source Tracing of Audio Deepfake Systems*. The approach leverages a pretrained countermeasure model (MiO) for feature extraction and performs separate classification tasks for acoustic models and vocoders. This repository processes audio data from the ASVspoof 2019 and MLAAD datasets to accomplish these tasks.

## Repository Structure

- **`mlaad_dataset_preprocess.py`**: Script for preprocessing the MLAAD dataset. It generates simplified metadata files (`simpler_meta.csv`) for each model directory in MLAAD, which contain essential columns: file path, model name, and architecture.
- **`thesis.py`**: Contains the MiO model training code for binary classification on the ASVspoof 2019 dataset.
- **`two_stage_process.py`**: Main script for implementing the two-stage process:
  - **Stage 1**: Trains the MiO model on ASVspoof 2019.
  - **Stage 2**: Uses the trained MiO model as a frozen feature extractor on MLAAD data, then trains separate classifiers for acoustic model and vocoder classification.
- **`whisper_feature_extractor.py`**: Contains functions for extracting CNN-based features from the Whisper model for the ASVspoof 2019 dataset.
- **`xlsr_feature_extractor.py`**: Contains functions for extracting CNN-based features from the XLS-R model for the ASVspoof 2019 dataset.

## Overview of the Two-Stage Process

### Stage 1: Training MiO Model on ASVspoof 2019 Dataset
The MiO model is initially trained on the ASVspoof 2019 dataset to perform binary classification (bonafide vs. spoof). After training, the MiO model is saved and frozen for feature extraction in Stage 2.

### Stage 2: Component Classification on MLAAD Dataset
Using the frozen MiO model, embeddings are generated for MLAAD dataset audio files. Two separate lightweight classification heads (fully connected neural networks) are then trained on these embeddings:
- **Acoustic Classifier**: Classifies based on acoustic models.
- **Vocoder Classifier**: Classifies based on vocoders.

## Files and Functions

### `mlaad_dataset_preprocess.py`
This script traverses the MLAAD dataset directory structure and generates `simpler_meta.csv` files for each model directory. The CSV contains:
- `file_path`: Path to each audio file.
- `model`: Model name used.
- `architecture`: Acoustic model architecture.

### `thesis.py`
This script trains the MiO model for binary spoof detection using ASVspoof 2019 data. The trained MiO model is then saved and can be loaded later for feature extraction.

### `two_stage_process.py`
This is the main script for the two-stage classification process:
- **`train_and_save_mio_model()`**: Trains and saves the MiO model if it hasn’t been trained yet.
- **`load_frozen_mio_model()`**: Loads the MiO model and freezes its layers to act as a feature extractor.
- **`extract_features_with_mio()`**: Extracts features from MLAAD audio samples using the frozen MiO model.
- **`MLAADDataset`**: Custom dataset class for loading MLAAD metadata, retrieving audio samples, and extracting embeddings.
- **`train_classification_head()`**: Function for training a lightweight classifier head on extracted features.
- **`evaluate_classification_head()`**: Evaluates classifier performance using accuracy, precision, recall, and F1 score.

### `whisper_feature_extractor.py` and `xlsr_feature_extractor.py`
These files extract CNN-based feature embeddings for audio samples using Whisper and XLS-R models, respectively. These embeddings are then used as input features in the MiO model.

## Usage

### 1. Prerequisites
Ensure you have the required dependencies:
- `torch`
- `torchaudio`
- `transformers`
- `pandas`
- `scikit-learn`

### 2. Data Preparation
Place ASVspoof 2019 and MLAAD datasets in directories accessible to the scripts. For MLAAD, ensure `meta.csv` files are correctly formatted in each model directory.

### 3. Running Stage 1: Train MiO Model on ASVspoof 2019
Run `train_and_save_mio_model()` in `two_stage_process.py` to train the MiO model on ASVspoof data for binary classification. The trained model will be saved for later use.

### 4. Running Stage 2: Feature Extraction and Classification with MLAAD
Load the MLAAD dataset and extract MiO embeddings with `extract_features_with_mio()` in the `MLAADDataset` class. Train and evaluate acoustic and vocoder classifiers using `train_classification_head()` and `evaluate_classification_head()`.

### Example Commands

```python
# Stage 1: Train MiO on ASVspoof 2019
train_and_save_mio_model()

# Load Frozen MiO Model for Feature Extraction
mio_model = load_frozen_mio_model()

# Prepare Dataloaders for Acoustic and Vocoder Classification
train_loader_acoustic = DataLoader(
    MLAADDataset(train_df, mlaad_root_dir, mio_model, label_type="acoustic"),
    batch_size=16, shuffle=True
)
eval_loader_acoustic = DataLoader(
    MLAADDataset(eval_df, mlaad_root_dir, mio_model, label_type="acoustic"),
    batch_size=16, shuffle=False
)

# Train and Evaluate Acoustic Classifier
train_classification_head(acoustic_classifier, train_loader_acoustic, eval_loader_acoustic)

# Repeat for Vocoder Classification
train_loader_vocoder = DataLoader(
    MLAADDataset(train_df, mlaad_root_dir, mio_model, label_type="vocoder"),
    batch_size=16, shuffle=True
)
eval_loader_vocoder = DataLoader(
    MLAADDataset(eval_df, mlaad_root_dir, mio_model, label_type="vocoder"),
    batch_size=16, shuffle=False
)
train_classification_head(vocoder_classifier, train_loader_vocoder, eval_loader_vocoder)
```
### RESULTS

Will be updated later