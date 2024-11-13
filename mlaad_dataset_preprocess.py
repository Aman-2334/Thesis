import os
import pandas as pd

# Set the root directory of the MLAAD dataset
root_dir = "Dataset\mlaad\MLAADv3\\fake"  # Replace with the actual path to the 'fake' folder

tts_model_vocoder_map = {
    # Facebook MMS-TTS models typically use Tacotron or Transformer-based architectures with HiFi-GAN vocoder
    "facebook/mms-tts-deu": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-eng": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-fin": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-fra": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-hun": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-nld": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-ron": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-rus": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-swe": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},
    "facebook/mms-tts-ukr": {"acoustic_model": "Transformer-based TTS", "vocoder": "HiFi-GAN"},

    # Griffin-Lim vocoder
    "griffin_lim": {"acoustic_model": "Unknown", "vocoder": "Griffin-Lim"},

    # Microsoft SpeechT5 uses Tacotron-style model with HiFi-GAN vocoder
    "microsoft/speecht5_tts": {"acoustic_model": "Tacotron", "vocoder": "HiFi-GAN"},

    # Parler TTS usually uses FastSpeech or Tacotron with HiFi-GAN or MelGAN
    "parler_tts": {"acoustic_model": "FastSpeech", "vocoder": "HiFi-GAN"},

    # Suno Bark is an all-in-one model with an internal vocoder
    "suno/bark": {"acoustic_model": "Suno Bark Model", "vocoder": "Internal"},

    # Coqui TTS models based on VITS with HiFi-GAN
    "tts_models/bg/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/cs/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/da/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/de/css10/vits-neon": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/de/thorsten/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/el/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/es/css10/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/et/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/fi/css10/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/fr/css10/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/ga/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/hu/css10/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/mt/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/nl/css10/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/pl/mai_female/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/ro/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/sk/cv/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/uk/mai/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},

    # Other models with specific acoustic/vocoder configurations
    "tts_models/en/blizzard2013/capacitron-t2-c50": {"acoustic_model": "Capacitron", "vocoder": "WaveGlow"},
    "tts_models/en/ek1/tacotron2": {"acoustic_model": "Tacotron2", "vocoder": "WaveGlow"},
    "tts_models/en/jenny/jenny": {"acoustic_model": "Jenny", "vocoder": "WaveNet"},
    "tts_models/en/ljspeech/fast_pitch": {"acoustic_model": "FastPitch", "vocoder": "WaveGlow"},
    "tts_models/en/ljspeech/glow-tts": {"acoustic_model": "Glow-TTS", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/neural_hmm": {"acoustic_model": "Neural HMM", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/overflow": {"acoustic_model": "Overflow", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/tacotron2-DCA": {"acoustic_model": "Tacotron2", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/tacotron2-DDC": {"acoustic_model": "Tacotron2", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/tacotron2-DDC_ph": {"acoustic_model": "Tacotron2", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/en/ljspeech/vits--neon": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/en/multi-dataset/tortoise-v2": {"acoustic_model": "Tortoise", "vocoder": "HiFi-GAN"},
    "tts_models/en/sam/tacotron-DDC": {"acoustic_model": "Tacotron", "vocoder": "WaveGlow"},
    "tts_models/fr/mai/tacotron2-DDC": {"acoustic_model": "Tacotron2", "vocoder": "HiFi-GAN"},
    "tts_models/it/mai_female/glow-tts": {"acoustic_model": "Glow-TTS", "vocoder": "HiFi-GAN"},
    "tts_models/it/mai_female/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/it/mai_male/glow-tts": {"acoustic_model": "Glow-TTS", "vocoder": "HiFi-GAN"},
    "tts_models/it/mai_male/vits": {"acoustic_model": "VITS", "vocoder": "HiFi-GAN"},
    "tts_models/multilingual/multi-dataset/xtts_v1.1": {"acoustic_model": "X-TTS", "vocoder": "HiFi-GAN"},
    "tts_models/multilingual/multi-dataset/xtts_v2": {"acoustic_model": "X-TTS", "vocoder": "HiFi-GAN"},
    "tts_models/uk/mai/glow-tts": {"acoustic_model": "Glow-TTS", "vocoder": "HiFi-GAN"},
}

# Initialize an empty set to store unique model names
unique_models = set()
unique_architecture = set()

# Function to process each 'simpler_meta.csv' file and collect unique models
def collect_unique_models(model_dir):
    # Path to 'simpler_meta.csv' file
    simpler_meta_path = os.path.join(model_dir, 'simpler_meta.csv')
    
    # Check if 'simpler_meta.csv' exists
    if os.path.isfile(simpler_meta_path):
        # Load simpler_meta.csv into a DataFrame
        simpler_meta_df = pd.read_csv(simpler_meta_path)
        
        # Collect unique models from the 'model' column
        unique_models.update(simpler_meta_df['model'].unique())
    else:
        print(f"'simpler_meta.csv' not found in {model_dir}")

def collect_unique_architecture(model_dir):
    # Path to 'simpler_meta.csv' file
    simpler_meta_path = os.path.join(model_dir, 'simpler_meta.csv')
    
    # Check if 'simpler_meta.csv' exists
    if os.path.isfile(simpler_meta_path):
        # Load simpler_meta.csv into a DataFrame
        simpler_meta_df = pd.read_csv(simpler_meta_path)
        
        # Collect unique models from the 'model' column
        unique_architecture.update(simpler_meta_df['architecture'].unique())
    else:
        print(f"'simpler_meta.csv' not found in {model_dir}")

def delete_simpler_meta_files():
    if os.path.isfile("unique_models.txt"):
        os.remove("unique_models.txt")
    if os.path.isfile("unique_architecture.txt"):
        os.remove("unique_architecture.txt")
    for language_dir in os.listdir(root_dir):
        language_path = os.path.join(root_dir, language_dir)
        if os.path.isdir(language_path):
            for model_dir in os.listdir(language_path):
                model_path = os.path.join(language_path, model_dir)
                if os.path.isdir(model_path):
                    # Path to simpler_meta.csv
                    simpler_meta_path = os.path.join(model_path, 'simpler_meta.csv')
                    # Check if the file exists, and delete if it does
                    if os.path.isfile(simpler_meta_path):
                        os.remove(simpler_meta_path)
                        print(f"Deleted: {simpler_meta_path}")
                    else:
                        print(f"'simpler_meta.csv' not found in {model_path}")

def update_simpler_meta(file_path):
    try:
        simpler_meta_df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return
    except pd.errors.EmptyDataError:
        print(f"Empty file or failed to load: {file_path}")
        return
    
    # Add new columns based on the 'model' column
    simpler_meta_df['vocoder'] = simpler_meta_df['model'].apply(
        lambda model: tts_model_vocoder_map.get(model, {}).get("vocoder", "Unknown")
    )
    simpler_meta_df['acoustic'] = simpler_meta_df['model'].apply(
        lambda model: tts_model_vocoder_map.get(model, {}).get("acoustic_model", "Unknown")
    )
    
    # Save the updated simpler_meta.csv
    simpler_meta_df.to_csv(file_path, index=False)
    print(f"Updated file saved: {file_path}")

def process_all_simpler_meta_files(root_dir):
    for language_dir in os.listdir(root_dir):
        language_path = os.path.join(root_dir, language_dir)
        if os.path.isdir(language_path):
            for model_dir in os.listdir(language_path):
                model_path = os.path.join(language_path, model_dir)
                if os.path.isdir(model_path):
                    # Path to simpler_meta.csv in this model directory
                    simpler_meta_path = os.path.join(model_path, 'simpler_meta.csv')
                    if os.path.isfile(simpler_meta_path):
                        update_simpler_meta(simpler_meta_path)
                    else:
                        print(f"'simpler_meta.csv' not found in {model_path}")
# Function to process each 'model_L_K' directory
def process_model_directory(model_dir):
    # Path to the existing 'meta.csv' file
    meta_file_path = os.path.join(model_dir, 'meta.csv')
    # Check if 'meta.csv' exists
    if os.path.isfile(meta_file_path):
        # Load meta.csv into a DataFrame
        meta_df = pd.read_csv(meta_file_path, delimiter='|')
        
        # Create the simpler meta DataFrame
        simpler_meta_df = meta_df[['path', 'model_name', 'architecture']].copy()
        simpler_meta_df.columns = ['file_path', 'model', 'architecture']  # Rename columns
        
        # Path to save the simpler meta file
        simpler_meta_path = os.path.join(model_dir, 'simpler_meta.csv')
        
        # Save the simpler meta DataFrame to CSV
        simpler_meta_df.to_csv(simpler_meta_path, index=False)
        print(f"Simpler metadata saved to: {simpler_meta_path}")
    else:
        print(f"'meta.csv' not found in {model_dir}")
# Traverse the directory structure
def create_simpler_meta_files():
    for language_dir in os.listdir(root_dir):
        language_path = os.path.join(root_dir, language_dir)
        if os.path.isdir(language_path):
            for model_dir in os.listdir(language_path):
                model_path = os.path.join(language_path, model_dir)
                if os.path.isdir(model_path):
                    # Process each model directory to create simpler_meta.csv
                    process_model_directory(model_path)
                    # Collect unique models from each model directory
                    collect_unique_models(model_path)
                    collect_unique_architecture(model_path)

    # Convert the set to a sorted list
    unique_model_list = sorted(unique_models)
    unique_architecture_list = sorted(unique_architecture)
    # Save the unique model list to a file
    unique_model_output_file = "unique_models.txt"
    with open(unique_model_output_file, "w") as f:
        for model in unique_model_list:
            f.write(model + "\n")
    unique_architecture_output_file = "unique_architecture.txt"
    with open(unique_architecture_output_file, "w") as f:
        for architecture in unique_architecture_list:
            f.write(architecture + "\n")
    print("Unique models have been saved to:", unique_model_output_file)
    print("Unique models have been saved to:", unique_architecture_output_file)
    process_all_simpler_meta_files(root_dir)

# create_simpler_meta_files()
# delete_simpler_meta_files()