import os  # 운영 체제와 상호 작용하기 위한 라이브러리
import librosa  # 오디오 파일에서 음악 정보를 추출하는 라이브러리
import numpy as np  # 배열 및 행렬 연산을 위한 라이브러리
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
# 머신러닝 모델 학습 및 검증을 위한 라이브러리
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler  # 데이터 전처리를 위한 라이브러리
import xgboost as xgb  # XGBoost 알고리즘을 사용한 머신러닝 모델 학습을 위한 라이브러리
from sklearn.metrics import accuracy_score  # 모델 성능 평가를 위한 라이브러리
from sklearn.preprocessing import LabelEncoder  # 데이터 전처리를 위한 라이브러리


def extract_features(file_path):
    # 파일 경로에서 오디오 데이터와 샘플 레이트를 로드한다. mono=True를 통해 스테레오 오디오를 모노로 변환한다.
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True)

    # 오디오 데이터에서 MFCC(Mel-Frequency Cepstral Coefficients)를 추출하고, 시간 축에서 평균을 계산한다.
    mfccs = np.mean(librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)

    # 오디오 데이터에서 Chroma feature를 추출하고, 시간 축에서 평균을 계산한다.
    chroma = np.mean(librosa.feature.chroma_stft(
        y=audio_data, sr=sample_rate).T, axis=0)

    # 오디오 데이터에서 Mel-scaled 스펙트로그램을 추출하고, 시간 축에서 평균을 계산한다.
    mel = np.mean(librosa.feature.melspectrogram(
        y=audio_data, sr=sample_rate).T, axis=0)

    # 오디오 데이터에서 스펙트럼 대비(Spectral Contrast)를 추출하고, 시간 축에서 평균을 계산한다.
    contrast = np.mean(librosa.feature.spectral_contrast(
        y=audio_data, sr=sample_rate).T, axis=0)

    # 오디오 데이터에서 Tonnetz feature를 추출하고, 시간 축에서 평균을 계산한다.
    tonnetz = np.mean(librosa.feature.tonnetz(
        y=librosa.effects.harmonic(audio_data), sr=sample_rate).T, axis=0)

    # 추출된 모든 특성들을 수평으로 연결(hstack)하고, 앞의 58개 특성만을 반환한다.
    return np.hstack([mfccs, chroma, mel, contrast, tonnetz])[:58]


# 오디오 파일의 특징을 저장한 CSV 파일을 읽어온다.
data = pd.read_csv("4.xgboost/Data/features_30_sec.csv")

# 음악 장르 레이블을 숫자로 변환한다.
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

# 'filename'과 'label' 열을 제외한 모든 열을 독립 변수 X에 할당
X = data.drop(['filename', 'label'], axis=1).values

# 'label' 열을 종속 변수 y에 할당
y = data['label'].values

# 독립 변수와 종속 변수를 훈련 데이터와 테스트 데이터로 분할 (테스트 데이터는 전체 데이터의 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 데이터를 정규화하기 위한 StandardScaler 객체 생성
scaler = StandardScaler()

# 훈련 데이터를 기준으로 스케일링을 학습하고, 훈련 데이터를 변환
X_train = scaler.fit_transform(X_train)

# 테스트 데이터를 동일한 스케일링 기준으로 변환
X_test = scaler.transform(X_test)

# XGBoost 분류기 객체 생성
# random_state는 일관된 결과를 얻기 위해 사용되며, 여기서는 42로 설정합니다.
xgb_model = xgb.XGBClassifier(random_state=42)

# 하이퍼파라미터 탐색 범위 설정
# n_estimators: 트리의 개수로, 앙상블에 사용되는 개별 결정 트리의 개수를 결정합니다.
# max_depth: 결정 트리의 최대 깊이로, 이를 통해 모델의 복잡성을 조절합니다.
# min_child_weight: 결정 트리에서 각 리프 노드의 데이터 가중치의 최소 합계로, 과적합을 방지하는 데 도움이 됩니다.
# learning_rate: 부스팅 스텝에 적용되는 학습률로, 이를 통해 각 트리가 학습하는 비율을 조절합니다.
param_grid = {'n_estimators': [10, 50, 100, 200],  # 사용할 트리 개수
              'max_depth': [None, 10, 20, 30],  # 트리 최대 깊이
              'min_child_weight': [1, 2, 4],  # 각 노드의 가중치에 대한 최소 합계
              'learning_rate': [0.1, 0.01, 0.001]}  # 학습률
# GridSearchCV 객체 생성
# xgb_model: XGBoost 분류기 객체
# param_grid: 탐색할 하이퍼파라미터 범위
# cv: 교차 검증을 수행할 때 사용할 폴드의 개수, 여기서는 5겹 교차 검증을 사용합니다.
# scoring: 평가 기준으로 사용할 메트릭, 여기서는 정확도(accuracy)를 사용합니다.
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')

# 그리드 탐색을 사용하여 모델 학습
# 이 과정에서 가능한 모든 하이퍼파라미터 조합에 대해 교차 검증을 수행하고, 가장 좋은 조합을 선택합니다.
grid_search.fit(X_train, y_train)


# 그리드 탐색에서 찾은 최적의 하이퍼파라미터를 가져옵니다.
best_params = grid_search.best_params_
print(f"최적의 하이퍼파라미터: {best_params}")

# 최적의 하이퍼파라미터를 사용하여 새로운 XGBoost 분류기 객체를 생성합니다.
# random_state는 일관된 결과를 얻기 위해 사용되며, 여기서는 42로 설정합니다.
xgb_model = xgb.XGBClassifier(**best_params, random_state=42)

# 최적의 하이퍼파라미터로 훈련된 XGBoost 분류기를 사용하여 훈련 데이터에 대해 학습을 진행합니다.
xgb_model.fit(X_train, y_train)

# 테스트할 음악 파일의 경로를 설정합니다.
test_file_path = "4.xgboost/Data/genres_original/blues/blues.00000.wav"

# 테스트 파일에서 특성을 추출하는 함수 extract_features를 호출하여 테스트 파일의 특성을 가져옵니다.
test_file_features = extract_features(test_file_path)


# 테스트 파일에서 추출한 특성을 스케일링하기 위해 앞서 훈련 데이터에 적용했던 StandardScaler 객체를 사용합니다.
test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))

# 스케일링된 테스트 파일 특성을 사용하여 장르를 예측합니다.
predicted_genre = xgb_model.predict(test_file_features_scaled)

# LabelEncoder 객체를 사용하여 예측된 장르의 정수 레이블을 문자열로 변환합니다.
predicted_genre_str = le.inverse_transform(predicted_genre)
print(f"예측한 음악 장르: {predicted_genre_str[0]}")

# 테스트 데이터에 대한 예측을 수행합니다.
y_pred = xgb_model.predict(X_test)

# 테스트 데이터에 대한 정확도를 계산합니다.
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy * 100:.2f}%")

# 교차 검증을 사용하여 훈련 데이터에 대한 모델 성능을 평가합니다.
# 여기서는 5-겹 교차 검증을 사용하며, 즉 훈련 데이터를 5개의 부분 집합으로 나누어 각각 테스트합니다.
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f"교차 검증된 정확도: {np.mean(cv_scores) * 100:.2f}%")
