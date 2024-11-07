
# Thesis

Keeping track of my thesis

## Venv Setup
pip install torch soundfile transformers pydub

torchaudio wasnt able to handle flac files so using soundfile instead

## Whisper Feature Extractor
- Dataset Class (ASVspoofWhisperDataset): 
  - Loads audio files and converts them to a fixed-length mel-spectrogram.
  - Pads or truncates each mel-spectrogram to ensure compatibility with Whisper’s expected input size (target_length = 3000).

- Whisper Feature Extraction:
  - For each batch, it extracts the last hidden state from the Whisper model, which serves as the feature representation.
  - whisper_features contains the feature embeddings of shape [batch_size, seq_length, hidden_size], where hidden_size is specific to the Whisper model variant (e.g., 768 for whisper-base).

- Inspecting Features:
  - Prints the shapes of the features and labels for verification.
  - Processes only one batch for quick inspection.

- Notes
  - Adjust Paths: Replace audio_dir and protocol_path with the correct paths for your dataset.
  - Adjust Model Variants: If you’re using a different Whisper variant, check and adjust the hidden size in subsequent processing steps.
This code should provide you with Whisper features from the ASVspoof2019 dataset, suitable for further processing or model training.

## XLS-R Feature Extractor
- XLS-R Model and Processor:
  - We load Wav2Vec2Model and Wav2Vec2Processor with the facebook/wav2vec2-xls-r-1b model.
- ASVspoof Dataset Class (ASVspoofXLSRDataset):
  - Loads audio files, resamples them if necessary, and processes them using xlsr_processor.
  - xlsr_processor converts waveforms to input values compatible with XLS-R, which it uses to generate embeddings.
- Feature Extraction:
  - xlsr_model(input_values) computes the last hidden state of XLS-R’s encoder, which is saved in xlsr_features.
- Confirm Output Shape:
  - The expected shape for xlsr_features should be [batch_size, seq_length, hidden_size], where hidden_size is specific to the XLS-R model (e.g., 1024 for wav2vec2-xls-r-1b).
- Expected Output
  - XLS-R Encoder Feature Shape: You should see a shape like [32, seq_length, 1024], where:
    - 32 is the batch size.
    - seq_length depends on the length of input audio, as XLS-R does not downsample as aggressively as Whisper.
    - 1024 is the hidden size of XLS-R’s xls-r-1b variant.
This code will provide you with the XLS-R features for each batch of audio from the ASVspoof2019 dataset.

Because the input audio tensors in the batch have different lengths, which makes it impossible to stack them directly. Since audio samples can vary in length, it’s common to apply padding so that all tensors in a batch have the same size. To handle this, we can use a custom collate_fn to pad each audio sample to the length of the longest sample in the batch.