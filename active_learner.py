import pandas as pd
import numpy as np
import librosa
import joblib
import os
import sys
import warnings
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
CSV_FILE = "my_10sec_segmented_dataset.csv"
MODEL_FILE = "deepfake_detector_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_COLS_FILE = "feature_columns.pkl"

TARGET_SR = 16000
TARGET_DURATION = 10
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

# --- LOAD RESOURCES ---
def load_resources():
    print("⏳ Loading model resources...")
    try:
        model = joblib.load(MODEL_FILE)
        le = joblib.load(LABEL_ENCODER_FILE)
        feature_cols = joblib.load(FEATURE_COLS_FILE)
        
        if os.path.exists(CSV_FILE):
            df = pd.read_csv(CSV_FILE)
        else:
            df = pd.DataFrame(columns=feature_cols + ['LABEL', 'filename'])
            
        return model, df, le, feature_cols
    except Exception as e:
        print(f"❌ Critical Error: {e}")
        print("👉 Run 'train_model.py' first!")
        sys.exit()

# --- FEATURE EXTRACTION ---
def extract_features_segment(y_segment, sr):
    features = {
        'chroma_stft': np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr)),
        'rms': np.mean(librosa.feature.rms(y=y_segment)),
        'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=y_segment, sr=sr)),
        'spectral_bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)),
        'rolloff': np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr)),
        'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(y_segment))
    }
    mfcc = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=20)
    for i in range(20):
        features[f'mfcc{i+1}'] = np.mean(mfcc[i])
    return features

def get_audio_features(file_path, label_str):
    try:
        # M4A/MP3 Support requires FFmpeg on Windows
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"\n❌ Error reading audio: {e}")
        if "NoBackendError" in str(e) or "audioread" in str(e):
            print("👉 TIP: For .m4a/.mp3 files, you need to install FFmpeg.")
        return []
    
    new_rows = []
    num_segments = int(np.ceil(len(y) / TARGET_SAMPLES))
    
    for i in range(num_segments):
        start = i * TARGET_SAMPLES
        end = min((i + 1) * TARGET_SAMPLES, len(y))
        y_seg = y[start:end]
        
        if len(y_seg) < TARGET_SAMPLES:
            y_seg = np.pad(y_seg, (0, TARGET_SAMPLES - len(y_seg)), 'constant')

        feats = extract_features_segment(y_seg, sr)
        feats['LABEL'] = label_str
        feats['filename'] = f"learn_{os.path.basename(file_path)}_{i}"
        new_rows.append(feats)
        
    return new_rows

# --- RE-TRAINING FUNCTION ---
def update_model_brain(file_path, true_label, df, le, feature_cols, model):
    print(f"📝 Adding data as '{true_label}' and retraining...")
    
    # 1. Extract Features with the TRUE label
    new_rows = get_audio_features(file_path, true_label)
    if not new_rows: return df, model # Stop if audio load failed

    # 2. Add to DataFrame
    new_df_chunk = pd.DataFrame(new_rows)
    df = pd.concat([df, new_df_chunk], ignore_index=True)
    
    # 3. Save CSV (Handle Excel Lock)
    try:
        df.to_csv(CSV_FILE, index=False)
        print("   ✅ Dataset CSV saved.")
    except PermissionError:
        print("\n   ❌ CRITICAL ERROR: File is locked!")
        print(f"   👉 Close '{CSV_FILE}' in Excel immediately.")
        input("   👉 Press Enter once closed to continue...")
        try:
            df.to_csv(CSV_FILE, index=False)
            print("   ✅ Saved on second try.")
        except:
            print("   ❌ Failed. Proceeding with in-memory update only.")

    # 4. Retrain
    print("   🧠 Retraining Random Forest...")
    X = df[feature_cols] # Force correct column order
    y = df['LABEL']
    y_enc = le.transform(y)
    
    model.fit(X, y_enc)
    joblib.dump(model, MODEL_FILE)
    print("   🚀 Model upgraded!")
    
    return df, model

# --- MAIN LOOP ---
def main():
    print("--- 🧠 ACTIVE LEARNING (POSITIVE & NEGATIVE REINFORCEMENT) ---")
    model, df, le, feature_cols = load_resources()
    
    classes = list(le.classes_)
    print(f"ℹ️  Classes: {classes}")
    fake_idx = classes.index("FAKE")
    real_idx = classes.index("REAL")

    while True:
        file_path = input("\nEnter path to audio file (or 'q' to quit): ").strip().replace('"', '')
        if file_path.lower() == 'q': break
        if not os.path.exists(file_path):
            print("❌ File not found.")
            continue

        # 1. PREDICT
        new_data = get_audio_features(file_path, "UNKNOWN")
        if not new_data: continue

        temp_df = pd.DataFrame(new_data)
        
        try:
            X_temp = temp_df[feature_cols]
        except KeyError as e:
             print(f"❌ Feature Mismatch. Re-run train_model.py. Error: {e}")
             continue

        probs_all = model.predict_proba(X_temp)
        avg_probs = np.mean(probs_all, axis=0)
        
        prob_fake = avg_probs[fake_idx]
        prob_real = avg_probs[real_idx]

        if prob_fake > prob_real:
            prediction = "FAKE"
            confidence = prob_fake
        else:
            prediction = "REAL"
            confidence = prob_real
        
        print(f"\n🤖 Prediction: {prediction} ({confidence*100:.1f}%)")
        
        # 2. ASK USER
        feedback = input("Is this correct? (y/n): ").lower()
        
        if feedback == 'n':
            # CORRECTION: It was wrong, teach it the opposite
            correct_label = "REAL" if prediction == "FAKE" else "FAKE"
            print(f"📉 Understood. Correcting mistake...")
            df, model = update_model_brain(file_path, correct_label, df, le, feature_cols, model)
            
        elif feedback == 'y':
            # REINFORCEMENT: It was right, teach it this file is a good example
            print(f"📈 Great! Reinforcing this knowledge...")
            df, model = update_model_brain(file_path, prediction, df, le, feature_cols, model)
            
        else:
            print("Invalid input. Skipping learning.")

if __name__ == "__main__":
    main()