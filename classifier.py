import os
import glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

DATASET_PATH = "data"  
SAMPLE_RATE = 22050   
DURATION = 200
N_MFCC = 13           

def load_audio_file(file_path, duration=DURATION, sr=SAMPLE_RATE):
    y, original_sr = librosa.load(file_path, sr=sr, mono=True)
    max_samples = sr * duration
    if len(y) > max_samples:
        y = y[:max_samples]
    else:
        y = np.hstack((y, np.zeros(max_samples - len(y))))
    
    return y, sr

def extract_features(y, sr, n_mfcc=N_MFCC):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc, axis=1)

def load_dataset(dataset_path=DATASET_PATH):
    X, y_labels = [], []
    genres = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        audio_files = glob.glob(os.path.join(genre_path, "*.wav"))
        
        for file_path in tqdm(audio_files, desc=f"Processing {genre}", unit="file"):
            audio, sr = load_audio_file(file_path)
            features = extract_features(audio, sr)
            X.append(features)
            y_labels.append(genre)
    
    return np.array(X), np.array(y_labels)

def main():
    X, y = load_dataset(DATASET_PATH)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    while True:
        user_file = input("\nEnter the path to an audio file for classification (or press Enter to skip): ").strip()
        if user_file:
            try:
                audio, sr = load_audio_file(user_file)
                features = extract_features(audio, sr).reshape(1, -1)
                predicted_genre = clf.predict(features)
                print(f"The predicted genre for '{user_file}' is: {predicted_genre[0]}")
            except Exception as e:
                print("Sorry, an error occurred while processing your file:", e)
        else:
            break 
        
if __name__ == "__main__":
    main()
