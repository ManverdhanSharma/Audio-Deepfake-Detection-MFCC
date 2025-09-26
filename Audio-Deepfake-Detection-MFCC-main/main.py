import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        audio_data, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file {audio_path}: {e}")
        return None
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return np.mean(mfccs.T, axis=0)

def create_dataset(directory, label):
    X, y = [], []
    audio_files = glob.glob(os.path.join(directory, "*.wav")) + glob.glob(os.path.join(directory, "*.mp3"))
    for audio_path in audio_files:
        mfcc_features = extract_mfcc_features(audio_path)
        if mfcc_features is not None:
            X.append(mfcc_features)
            y.append(label)
        else:
            print(f"Skipping audio file {audio_path}")
    print("Number of samples in", directory, ":", len(X))
    print("Filenames in", directory, ":", [os.path.basename(path) for path in audio_files])
    return X, y

def train_model(X, y):
    unique_classes = np.unique(y)
    print("Unique classes in y_train:", unique_classes)
    if len(unique_classes) < 2:
        raise ValueError("At least 2 classes required to train")
    
    print("Size of X:", X.shape)
    print("Size of y:", y.shape)
    
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Training on all available data without test split.")
        X_train, y_train = X, y
        X_test, y_test = None, None
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print("Size of X_train:", X_train.shape)
        print("Size of X_test:", X_test.shape)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_scaled, y_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        y_pred = svm_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
    
    # Save the trained model
    joblib.dump(svm_classifier, "svm_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model saved successfully!")

def analyze_audio(input_audio_path):
    try:
        svm_classifier = joblib.load("svm_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return
    
    if not os.path.exists(input_audio_path):
        print("Error: File does not exist.")
        return
    
    mfcc_features = extract_mfcc_features(input_audio_path)
    if mfcc_features is not None:
        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))
        prediction = svm_classifier.predict(mfcc_features_scaled)
        
        if prediction[0] == 0:
            print(f"GENUINE (Real Human Voice)")
        else:
            print(f"DEEPFAKE (AI Generated)")
    else:
        print("Error: Unable to process the audio.")

def main():
    genuine_dir = "real_audio"
    deepfake_dir = "deepfake_audio"
    
    print("Creating dataset from audio files...")
    X_genuine, y_genuine = create_dataset(genuine_dir, label=0)
    X_deepfake, y_deepfake = create_dataset(deepfake_dir, label=1)
    
    if len(X_genuine) == 0:
        print("Error: No genuine audio files found!")
        return
    if len(X_deepfake) == 0:
        print("Error: No deepfake audio files found!")
        return
    
    X = np.vstack((X_genuine, X_deepfake))
    y = np.hstack((y_genuine, y_deepfake))
    
    print(f"Total samples: {len(X)} ({len(X_genuine)} genuine + {len(X_deepfake)} deepfake)")
    train_model(X, y)

if __name__ == "__main__":
    main()
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE!")
    print("="*50)
    
    while True:
        user_input_file = input("\nEnter audio file path to analyze (or 'quit'): ")
        if user_input_file.lower() in ['quit', 'exit', 'q']:
            break
        analyze_audio(user_input_file)
