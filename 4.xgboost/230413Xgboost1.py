import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def extract_features(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)
    mfccs = np.mean(librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(
        y=audio_data, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(
        y=audio_data, sr=sample_rate).T, axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio_data), sr=sample_rate).T, axis=0)
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])[:58]


data = pd.read_csv("4.xgboost/Data/features_30_sec.csv")

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.drop(['filename', 'label'], axis=1).values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

xgb_model = xgb.XGBClassifier(random_state=42)
param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_child_weight': [1, 2, 4],
              'learning_rate': [0.1, 0.01, 0.001]}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f"최적의 하이퍼파라미터: {best_params}")

xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)

test_file_path = "4.xgboost/Data/genres_original/blues/blues.00000.wav"

test_file_features = extract_features(test_file_path)

test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))

predicted_genre = xgb_model.predict(test_file_features_scaled)

predicted_genre_str = le.inverse_transform(predicted_genre)
print(f"예측한 음악 장르: {predicted_genre_str[0]}")

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy * 100:.2f}%")

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f"교차 검증된 정확도: {np.mean(cv_scores) * 100:.2f}%")
