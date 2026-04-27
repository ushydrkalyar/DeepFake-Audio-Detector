import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

# --- CONFIGURATION ---
CSV_FILE = "my_10sec_segmented_dataset.csv"
MODEL_FILENAME = "deepfake_detector_model.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

def train():
    print("--- 1. LOADING DATA ---")
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        print(f"❌ Error: {CSV_FILE} not found. Did you run generate_data.py?")
        return

    # 1. Prepare Features (X) and Target (y)
    # We drop 'LABEL' (target) and 'filename' (not a feature)
    X = df.drop(['LABEL', 'filename'], axis=1, errors='ignore')
    y = df['LABEL']

    # 2. Encode Labels (Real/Fake -> 0/1)
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    print(f"✅ Loaded {len(df)} samples.")
    print(f"   Features: {list(X.columns)}")
    print(f"   Classes: {list(le.classes_)}")

    # 3. Split Data (80% Train, 20% Test)
    print("\n--- 2. TRAINING MODEL ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and Train Random Forest
    # n_jobs=-1 uses all CPU cores for faster training
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("✅ Model training complete.")

    # 4. Evaluate
    print("\n--- 3. EVALUATION ---")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
    print("\nDetailed Report:")
    print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))

    # 5. Save Model & Tools
    print("\n--- 4. SAVING ARTIFACTS ---")
    joblib.dump(model, MODEL_FILENAME)
    joblib.dump(le, LABEL_ENCODER_FILENAME)
    joblib.dump(list(X.columns), "feature_columns.pkl")
    print(f"🎉 Saved '{MODEL_FILENAME}' and helper files.")

if __name__ == "__main__":
    train()