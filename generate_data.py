import os
import zipfile
import shutil
import time
import numpy as np
import pandas as pd
import librosa
import warnings

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
TARGET_SR = 16000           # 16 kHz sample rate
TARGET_DURATION = 10        # 10 seconds fixed duration
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION 

# Input Filenames (Must exist in the same folder)
REAL_ZIP_FILENAME = "realVoices.zip"
FAKE_ZIP_FILENAME = "fakeVoices.zip"

# Temporary extraction folders
REAL_DIR = "temp_real_audio"
FAKE_DIR = "temp_fake_audio"

# Output file
OUTPUT_CSV = "my_10sec_segmented_dataset.csv"


# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def setup_directories():
    """Cleans up old folders and creates new ones."""
    print("--- 1. SETTING UP DIRECTORIES ---")
    if os.path.exists(REAL_DIR): shutil.rmtree(REAL_DIR)
    if os.path.exists(FAKE_DIR): shutil.rmtree(FAKE_DIR)
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

def extract_zip(zip_name, target_dir):
    """Extracts a zip file to the target directory."""
    if not os.path.exists(zip_name):
        raise FileNotFoundError(f"❌ Error: '{zip_name}' not found. Please place it in this folder.")
    
    print(f"   Extracting '{zip_name}'...", end=" ", flush=True)
    try:
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("✅ Done.")
    except zipfile.BadZipFile:
        print("\n❌ Error: The file is corrupted.")
        raise

def extract_features_from_segment(y_segment, sr):
    """Calculates features for a single audio segment."""
    features = {
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr)),
        'rms': np.mean(librosa.feature.rms(y=y_segment)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y_segment))
    }
    
    # MFCCs
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}'] = np.mean(mfcc[i])
        
    return features

def process_file_segments(file_path, label_value, all_data):
    """Segments an audio file and extracts features."""
    try:
        # Load and Resample
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
        
        num_segments = len(y) // TARGET_SAMPLES
        segments_added = 0
        
        # 1. Process full segments
        for i in range(num_segments):
            y_segment = y[i * TARGET_SAMPLES : (i+1) * TARGET_SAMPLES]
            feats = extract_features_from_segment(y_segment, sr)
            if feats:
                feats['LABEL'] = label_value
                feats['filename'] = f"{os.path.basename(file_path).split('.')[0]}_seg{i}"
                all_data.append(feats)
                segments_added += 1

        # 2. Process remaining part (padded)
        if len(y) % TARGET_SAMPLES > 0:
            y_segment = y[num_segments * TARGET_SAMPLES:]
            padding = TARGET_SAMPLES - len(y_segment)
            y_segment = np.pad(y_segment, (0, padding), 'constant')
            
            feats = extract_features_from_segment(y_segment, sr)
            if feats:
                feats['LABEL'] = label_value
                feats['filename'] = f"{os.path.basename(file_path).split('.')[0]}_seg{num_segments}"
                all_data.append(feats)
                segments_added += 1
                
        return segments_added

    except Exception as e:
        # print(f"Error processing {file_path}: {e}") # Uncomment to debug specific files
        return 0

def process_directory(directory, label_value, data_list):
    """Walks through a directory and processes all audio files."""
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                file_list.append(os.path.join(root, file))

    total_files = len(file_list)
    print(f"\n📂 Processing {total_files} files in '{directory}'...")
    
    start_time = time.time()
    count = 0
    
    for i, file_path in enumerate(file_list):
        count += process_file_segments(file_path, label_value, data_list)
        
        # Terminal Progress Bar
        if (i + 1) % 10 == 0 or (i + 1) == total_files:
            elapsed = time.time() - start_time
            print(f"   Progress: {i+1}/{total_files} files | Segments Created: {count} | Time: {elapsed:.1f}s", end='\r')
            
    print(f"\n   ✅ Completed {label_value}. Total segments: {count}")
    return count

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    try:
        # 1. Setup
        setup_directories()
        extract_zip(REAL_ZIP_FILENAME, REAL_DIR)
        extract_zip(FAKE_ZIP_FILENAME, FAKE_DIR)
        
        # 2. Process
        print("\n--- 2. GENERATING DATASET (Feature Extraction) ---")
        print(f"Configuration: {TARGET_SR}Hz | {TARGET_DURATION}s Segments")
        
        all_data = []
        process_directory(REAL_DIR, 'REAL', all_data)
        process_directory(FAKE_DIR, 'FAKE', all_data)
        
        # 3. Save
        print("\n--- 3. SAVING DATASET ---")
        if len(all_data) > 0:
            df = pd.DataFrame(all_data)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"🎉 SUCCESS! Saved '{OUTPUT_CSV}' with {len(df)} rows.")
            
            # Cleanup
            print("🧹 Cleaning up temporary folders...")
            shutil.rmtree(REAL_DIR)
            shutil.rmtree(FAKE_DIR)
            print("✨ Done.")
        else:
            print("❌ Error: No audio segments were generated. Please check your audio files.")

    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        input("Press Enter to exit...") # Keeps window open if double-clicked