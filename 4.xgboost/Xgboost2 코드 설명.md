# Xgboost2 코드 설명



* `Xgboost1`에서 변경 한 부분만 설명하였습니다.



**변경 전 코드**

```python
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
```

**변경 후 코드**

```python
def extract_features(file_path, n_features):
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
    feature_vector = np.hstack([mfccs, chroma, mel, contrast, tonnetz])
    return feature_vector[:n_features]

n_features = X_train.shape[1]

test_file_features = extract_features(test_file_path, n_features)
```

* 이렇게 하면, 특성 개수가 변경되더라도 코드를 수정하지 않고도 자동으로 처리된다. 또한, 추가적으로 특성 선택 기법을 적용하여 더 중요한 특성만 선택하는 것도 고려할 수 있다. 이를 통해 더 간결한 특성 벡터를 얻어 학습 속도를 높이고, 모델의 일반화 성능을 향상시킬 수 있다.
* `feature_vector = np.hstack([mfccs, chroma, mel, contrast, tonnetz])`
  `return feature_vector[:n_features]`
  오디오 파일의 특성을 결합하고 반환되는 특성 벡터의 크기를 훈련 데이터셋과 동일하게 유지하기 위해 추가했다.
  이 코드는 오디오 파일에서 추출한 각 특성을 하나의 특성 벡터로 결합하고, 반환되는 특성 벡터의 크기를 `n_features`와 동일하게 제한하는 데 사용된다.
  `np.hstack()` 함수를 사용하여 `mfccs`, `chroma`, `mel`, `contrast`, `tonnetz` 등 다양한 특성을 하나의 넘파이 배열로 합친다. 이렇게 하면 각 오디오 파일의 특성을 모델에 전달하기 위한 단일 특성 벡터를 얻을 수 있다. 그런 다음 `return feature_vector[:n_features]` 코드를 사용하여 반환되는 특성 벡터의 크기를 제한한다. 이렇게 하면 훈련 데이터셋과 동일한 차원의 특성 벡터를 반환할 수 있다. 이렇게 하면 모델이 훈련 데이터셋과 동일한 차원의 입력 데이터를 받을 수 있어 예측 시 오류가 발생하는 것을 방지할 수 있다.
* `n_features = X_train.shape[1]` 
  훈련 데이터셋과 테스트 파일이 동일한 차원의 특성 벡터를 갖도록 하기 위해 추가했다.
  `n_features = X_train.shape[1]` 코드는 훈련 데이터셋의 특성 개수를 저장하는 데 사용된다. 이 값을 사용하여 테스트 파일의 특성을 추출할 때, 반환되는 특성 벡터가 훈련 데이터셋과 동일한 차원을 갖도록 할 수 있다. 여기서 `X_train.shape[1]`는 훈련 데이터셋 `X_train`의 열 개수를 나타냅니다. 이는 각 오디오 파일의 특성 개수와 같다. `n_features` 변수를 사용하여 `extract_features()` 함수에 전달하면, 함수는 반환되는 특성 벡터의 크기를 `n_features`와 동일하게 제한한다. 이렇게 하면 훈련 데이터셋과 테스트 파일이 동일한 차원의 특성 벡터를 갖게 되어 모델의 입력으로 사용할 수 있다.
* `test_file_features = extract_features(test_file_path, n_features)`
  테스트 파일의 특성을 추출, 이를 사용하여 모델을 테스트하는 데 필요한 작업을 수행하기 위해 추가했다. 여기서는 `test_file_path` 변수에 저장된 오디오 파일의 특성을 추출하고, 이 특성을 나중에 XGBoost 모델을 사용하여 장르를 예측하는 데 사용한다. `n_features` 변수는 특성 추출 함수인 `extract_features()`에 전달되어, 반환되는 특성 벡터의 크기를 제한하는 데 사용된다. 이를 통해 다른 데이터와 동일한 차원의 특성 벡터를 얻을 수 있다.

**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 2, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 75.50%
교차 검증된 정확도: 77.25%
```







**변경 전 코드**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

**변경 후 코드**

* `random_state`를 지정하지 않을 경우

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)
```

*  `random_state=42`를 제거하여 매번 실행할 때마다 다른 훈련 및 테스트 데이터셋을 얻는다.
* `random_state` 매개변수를 지정하면, `train_test_split` 함수는 데이터셋을 분할하기 전에 동일한 난수 시드를 사용하여 데이터를 섞는다. 이렇게 하면 코드를 여러 번 실행할 때마다 동일한 훈련 및 테스트 데이터셋이 생성되어 재현 가능한 결과를 얻을 수 있다. 그러나 `random_state`를 지정하지 않거나 `None`으로 설정하면, `train_test_split` 함수는 매번 실행할 때마다 다른 난수 시드를 사용하여 데이터를 섞는다. 이 경우 훈련 및 테스트 데이터셋이 매번 다르게 생성되므로, 모델이 데이터의 다양한 측면을 학습할 수 있는 기회를 제공한다. 
* 이 경우, 교차 검증을 사용하여 모델의 일반화 성능을 평가할 수 있다. 코드에 이미 교차 검증을 사용하고 있으므로, 이러한 변경이 적용되면 모델의 일반화 성능에 대한 더 신뢰할 수 있는 추정치를 얻을 수 있을 것이라 예상된다.

**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 4, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 77.00%
교차 검증된 정확도: 77.63%
```







**변경 전 코드**

```python
xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)
```

**변경 후 코드**

* `random_state`를 지정하지 않을 경우

```python
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train)
```

**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.1, 'max_depth': 10, 'min_child_weight': 4, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 77.00%
교차 검증된 정확도: 77.63%
```





**변경 전 코드**

```python
xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)
```

**변경 후 코드**

* `random_state`를 지정하지 않을 경우

```python
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train)
```

**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 4, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 82.50%
교차 검증된 정확도: 75.75%
```





**변경 전 코드**

```python
xgb_model = xgb.XGBClassifier()
param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_child_weight': [1, 2, 4],
              'learning_rate': [0.1, 0.01, 0.001]}
grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

**변경 후 코드( `RandomizedSearchCV`를 사용하도록 수정)**

```python
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(10, 300),
    'max_depth': [None] + list(range(10, 31)),
    'min_child_weight': randint(1, 5),
    'learning_rate': uniform(0.001, 0.2)
}

random_search = RandomizedSearchCV(
    xgb_model, param_dist, n_iter=100, cv=5, scoring='accuracy')
random_search.fit(X_train, y_train)

best_params = random_search.best_params_
print(f"최적의 하이퍼파라미터: {best_params}")
```

*  `RandomizedSearchCV`를 사용하여 하이퍼파라미터를 최적화했다. `param_dist` 변수에 정의된 분포를 사용하여 무작위 하이퍼파라미터 탐색을 진행하고, `n_iter` 변수를 통해 탐색할 하이퍼파라미터의 수를 지정한다. 이렇게 수정된 코드를 사용하면 더 다양한 하이퍼파라미터 조합을 탐색할 수 있어 더 좋은 결과를 얻을 가능성이 높아진다.

**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.136256087783457, 'max_depth': 24, 'min_child_weight': 2, 'n_estimators': 65}
예측한 음악 장르: classical
정확도: 81.00%
교차 검증된 정확도: 76.75%
```

**데이터 셋을 features_3_sec.csv로 변경한 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.09796599427179664, 'max_depth': None, 'min_child_weight': 3, 'n_estimators': 243}
예측한 음악 장르: classical
정확도: 90.14%
교차 검증된 정확도: 88.39%
```



