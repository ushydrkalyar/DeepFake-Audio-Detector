import os
import numpy as np
import pandas as pd
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURATION ---
MODEL_FILE = "deepfake_detector_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_COLS_FILE = "feature_columns.pkl"

TARGET_SR = 16000
TARGET_DURATION = 10
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

def extract_features_from_segment(y_segment, sr):
    """Exact same logic as generate_data.py"""
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

def predict_audio(file_path):
    # 1. Load Resources
    try:
        model = joblib.load(MODEL_FILE)
        le = joblib.load(LABEL_ENCODER_FILE)
        feature_cols = joblib.load(FEATURE_COLS_FILE)
    except FileNotFoundError:
        print("❌ Error: Model files not found. Run train_model.py first.")
        return

    print(f"\n🔍 Analyzing: {os.path.basename(file_path)}")
    
    # 2. Load Audio
    try:
        y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)
    except Exception as e:
        print(f"Error reading audio: {e}")
        return

    # 3. Segment & Predict
    num_segments = int(np.ceil(len(y) / TARGET_SAMPLES))
    segment_predictions = []
    
    print(f"   Audio Length: {len(y)/sr:.2f}s | Segments: {num_segments}")

    for i in range(num_segments):
        # Slice the segment
        start = i * TARGET_SAMPLES
        end = min((i + 1) * TARGET_SAMPLES, len(y))
        y_seg = y[start:end]

        # Pad if short (e.g., the last segment)
        if len(y_seg) < TARGET_SAMPLES:
            padding = TARGET_SAMPLES - len(y_seg)
            y_seg = np.pad(y_seg, (0, padding), 'constant')

        # Extract Features
        feats_dict = extract_features_from_segment(y_seg, sr)
        
        # Convert to DataFrame and reorder columns
        feat_df = pd.DataFrame([feats_dict])[feature_cols]
        
        # Predict Probability of being "Class 1" (usually Fake, but we check label encoder)
        # model.predict_proba returns [[prob_class_0, prob_class_1]]
        prob = model.predict_proba(feat_df)[0]
        segment_predictions.append(prob)

    # 4. Aggregate Results (Average the probabilities across all segments)
    avg_probs = np.mean(segment_predictions, axis=0)
    final_class_index = np.argmax(avg_probs)
    final_confidence = avg_probs[final_class_index]
    final_label = le.inverse_transform([final_class_index])[0]

    # 5. Display Result
    print("\n" + "="*40)
    print(f"🎤 RESULT:  {final_label.upper()}")
    print(f"📊 CONFIDENCE: {final_confidence * 100:.2f}%")
    print("="*40)
    
    # Optional: Detailed segment breakdown
    # print(f"   (Class 0: {avg_probs[0]:.2f}, Class 1: {avg_probs[1]:.2f})")

if __name__ == "__main__":
    # ASK USER FOR FILE
    path = input("Enter the path to your audio file (e.g., test.wav): ").strip().replace('"', '')
    if os.path.exists(path):
        predict_audio(path)
    else:
        print("❌ File not found.")