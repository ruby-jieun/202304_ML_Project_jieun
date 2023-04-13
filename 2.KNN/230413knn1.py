import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


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
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])[:58]  # 수정된 부분


# 데이터셋 불러오기
data = pd.read_csv("2.KNN/[230413]KNN/Data/features_30_sec.csv")

# 특성과 레이블 분리
X = data.drop(['filename', 'label'], axis=1).values
y = data['label'].values

# 데이터를 학습용과 테스트용으로 분리 (예: 80% 학습, 20% 테스트)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 특성 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN 분류기 생성 및 학습 (k값은 5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# 테스트 파일 경로 지정
test_file_path = "2.KNN/[230413]KNN/Data/genres_original/blues/blues.00000.wav"

# 테스트 파일에서 특성 추출
test_file_features = extract_features(test_file_path)

# 특성 스케일링
test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))

# 테스트 파일에 대한 장르 예측
predicted_genre = knn.predict(test_file_features_scaled)

# 예측 결과 출력
print(f"예측한 음악 장르: {predicted_genre[0]}")

# 정확도 평가
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy * 100:.2f}%")
