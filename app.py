import streamlit as st
import os
import numpy as np
import pandas as pd
import librosa
import joblib
import warnings

warnings.filterwarnings("ignore")

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🕵️", layout="centered")

# --- CONSTANTS ---
MODEL_FILE = "deepfake_detector_model.pkl"
LABEL_ENCODER_FILE = "label_encoder.pkl"
FEATURE_COLS_FILE = "feature_columns.pkl"
TARGET_SR = 16000
TARGET_DURATION = 10
TARGET_SAMPLES = TARGET_SR * TARGET_DURATION

# --- LOAD BRAIN (Cached for speed) ---
@st.cache_resource
def load_model_resources():
    try:
        model = joblib.load(MODEL_FILE)
        le = joblib.load(LABEL_ENCODER_FILE)
        cols = joblib.load(FEATURE_COLS_FILE)
        return model, le, cols
    except Exception as e:
        return None, None, None

# --- FEATURE EXTRACTOR ---
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

# --- UI LAYOUT ---
st.title("🕵️ Deepfake Audio Detector")
st.write("Upload an audio file to check if it's **Real** or **AI-Generated**.")

# 1. Load Model
model, le, feature_cols = load_model_resources()

if model is None:
    st.error("❌ Model files not found! Please run `train_model.py` first.")
    st.stop()

# 2. File Uploader (UPDATED: Added "opus")
uploaded_file = st.file_uploader("Choose an audio file...", type=["wav", "mp3", "m4a", "flac", "opus"])

if uploaded_file is not None:
    # Save temp file
    # We use the original extension to help librosa/ffmpeg detect the format
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_filename = f"temp_audio_check{file_extension}"
    
    with open(temp_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Audio Player
    try:
        st.audio(uploaded_file, format=f'audio/{file_extension.replace(".", "")}')
    except:
        st.warning("Could not play audio preview (format might be unsupported by browser), but analysis will still work.")
    
    if st.button("🔍 Analyze Audio"):
        with st.spinner("Listening and analyzing..."):
            try:
                # Load Audio
                # Librosa uses FFmpeg to handle .opus and .mp3
                y, sr = librosa.load(temp_filename, sr=TARGET_SR, mono=True)
                
                # Segment Logic
                num_segments = int(np.ceil(len(y) / TARGET_SAMPLES))
                segment_predictions = []
                
                # Progress Bar
                progress_bar = st.progress(0)
                
                for i in range(num_segments):
                    # Update bar
                    progress_bar.progress((i + 1) / num_segments)
                    
                    # Slice
                    start = i * TARGET_SAMPLES
                    end = min((i + 1) * TARGET_SAMPLES, len(y))
                    y_seg = y[start:end]
                    
                    # Pad
                    if len(y_seg) < TARGET_SAMPLES:
                        y_seg = np.pad(y_seg, (0, TARGET_SAMPLES - len(y_seg)), 'constant')
                    
                    # Extract & Predict
                    feats = extract_features_segment(y_seg, sr)
                    df_feat = pd.DataFrame([feats])[feature_cols] # Force column order
                    
                    probs = model.predict_proba(df_feat)[0]
                    segment_predictions.append(probs)

                # Aggregate
                avg_probs = np.mean(segment_predictions, axis=0)
                
                # Identify FAKE vs REAL index
                classes = list(le.classes_)
                fake_idx = classes.index("FAKE")
                real_idx = classes.index("REAL")
                
                prob_fake = avg_probs[fake_idx]
                prob_real = avg_probs[real_idx]

                # Final Decision
                if prob_fake > prob_real:
                    label = "FAKE"
                    confidence = prob_fake
                    color = "red"
                    icon = "🤖"
                else:
                    label = "REAL"
                    confidence = prob_real
                    color = "green"
                    icon = "👤"

                # Display Result
                st.markdown("---")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.header("Verdict")
                    st.markdown(f":{color}[**{icon} {label}**]")
                
                with col2:
                    st.header("Confidence")
                    st.metric(label="Model Certainty", value=f"{confidence*100:.1f}%")
                
                # Detailed Gauge
                st.write("### AI Probability Score")
                st.progress(int(prob_fake * 100))
                st.caption(f"0% = Definitely Real | 100% = Definitely AI. This file scored: {int(prob_fake*100)}%")

            except Exception as e:
                st.error(f"Error processing audio: {e}")
                if "NoBackendError" in str(e) or "audioread" in str(e):
                    st.warning("⚠️ You need FFmpeg installed to process .opus/.m4a/.mp3 files.")
                    st.info("Download it from gyan.dev/ffmpeg/builds, extract it, and put 'ffmpeg.exe' in this project folder.")
    
    # Cleanup
    if os.path.exists(temp_filename):
        os.remove(temp_filename)