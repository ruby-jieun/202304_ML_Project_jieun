import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
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
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])[:58]


# 데이터셋 불러오기
data = pd.read_csv("2.KNN/[230413]KNN/Data/features_30_sec.csv")

num_rows = data.shape[0]
print(f"데이터셋의 행 개수: {num_rows}")

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

# 랜덤 포레스트 분류기 생성 및 하이퍼파라미터 튜닝
# 랜덤 포레스트 분류기 객체를 생성. random_state 매개변수를 42로 설정하여 결과의 재현성을 확보
rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [10, 50, 100, 200],  # 랜덤 포레스트 내의 결정 트리 개수. [10, 50, 100, 200] 중에서 선택
              # 결정 트리의 최대 깊이. [None, 10, 20, 30] 중에서 선택. None은 무한대로 간주되어 최대 깊이 제한이 없음
              'max_depth': [None, 10, 20, 30],
              # 결정 트리의 노드에서 분할을 위한 최소 샘플 개수. [2, 5, 10] 중에서 선택
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}  # 결정 트리의 리프 노드에 필요한 최소 샘플 개수. [1, 2, 4] 중에서 선택
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
# GridSearchCV는 랜덤 포레스트 분류기(rf)를 사용하며, 지정된 하이퍼파라미터 범위(param_grid)에 대해 교차 검증(cross-validation, cv)을 수행, 최적의 하이퍼파라미터 조합을 찾는다.
# 교차 검증 폴드(cv)는 5개로 설정되어 있으며, 검증 점수(scoring)는 'accuracy'를 사용한다.
grid_search.fit(X_train, y_train)
# GridSearchCV 객체를 사용하여 하이퍼파라미터 조합에 대한 교차 검증을 수행한 후, 최적의 하이퍼파라미터를 찾기 위해 학습 데이터(X_train, y_train)에 적합시킴

# 최적의 하이퍼파라미터 출력
best_params = grid_search.best_params_
print(f"최적의 하이퍼파라미터: {best_params}")

# 최적의 하이퍼파라미터를 사용하여 랜덤 포레스트 분류기 생성 및 학습
rf = RandomForestClassifier(**best_params, random_state=42)
rf.fit(X_train, y_train)


# 테스트 파일 경로 지정
test_file_path = "2.KNN/[230413]KNN/Data/genres_original/blues/blues.00000.wav"

# 테스트 파일에서 특성 추출
test_file_features = extract_features(test_file_path)

# 특성 스케일링
test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))

# 테스트 파일에 대한 장르 예측
predicted_genre = rf.predict(test_file_features_scaled)

# 예측 결과 출력
print(f"예측한 음악 장르: {predicted_genre[0]}")

# 정확도 평가
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy * 100:.2f}%")

# 교차 검증으로 정확도 평가
cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"교차 검증된 정확도: {np.mean(cv_scores) * 100:.2f}%")
