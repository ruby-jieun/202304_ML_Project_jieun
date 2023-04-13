# Xgboost1 코드 설명







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

* 코드에서 `np.hstack([mfccs, chroma, mel, contrast, tonnetz])[:58]`를 사용하여 58개의 특성만 반환한 이유는, 원래 데이터셋에서 사용된 특성 개수와 일치시키기 위함이다.
  이렇게 하여 훈련된 모델과 테스트 데이터의 차원을 맞춰준다. 그러나 이 방법은 하드코딩된 58이라는 숫자에 의존하므로 좋은 방법이라고 할 수 없다.
* 더 좋은 방법은 원래 데이터셋에서 사용한 특성의 개수를 자동으로 결정하고, 그에 따라 반환되는 특성 개수를 조절하는 것이다. 이를 위해 데이터셋의 특성 개수를 계산한 다음, 이 값을 `extract_features` 함수에 전달하여 사용하면 된다.(`Xgboost2`에서 수정 예정)
* 그렇게 하면, 특성 개수가 변경되더라도 코드를 수정하지 않고도 자동으로 처리된다. 또한, 추가적으로 특성 선택 기법을 적용하여 더 중요한 특성만 선택하는 것도 고려할 수 있다. 이를 통해 더 간결한 특성 벡터를 얻어 학습 속도를 높이고, 모델의 일반화 성능을 향상시킬 수 있다.









```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
```

```python
xgb_model = xgb.XGBClassifier(**best_params, random_state=42)
xgb_model.fit(X_train, y_train)
```

```python
xgb_model = xgb.XGBClassifier(random_state=42)
```

* `random_state=42`로 설정한 이유는 재현 가능한 일관된 결과를 얻기 위함이다. 

* `train_test_split` 함수는 데이터를 무작위로 분할하기 때문에, `random_state` 매개변수를 설정함으로써 동일한 무작위 분할 결과를 항상 얻을 수 있다.
  이렇게 함으로써 다른 사람들이 코드를 실행했을 때 동일한 결과를 얻을 수 있어, 코드의 행동이 예측 가능하게 된다.
  
* 다른 접근 방식으로, `random_state`를 지정하지 않거나 다른 값을 사용할 수 있다. 이 경우, 교차 검증을 사용하여 모델의 일반화 성능을 평가할 수 있다.(교차 검증은 데이터를 여러 폴드로 나누고, 각 폴드를 테스트 세트로 사용하면서 모델을 반복적으로 학습 및 평가하는 방법이다.) 이렇게 하면, `random_state` 값을 고정시키지 않고도 모델의 일반화 성능에 대한 신뢰할 수 있는 추정치를 얻을 수 있다.(`Xgboost2`에서 수정 예정)

* `xgb_model = xgb.XGBClassifier(**best_params, random_state=42)`:

  XGBoost 분류기 객체를 생성한다. `**best_params`는 앞에서 GridSearchCV를 통해 찾은 최적의 하이퍼파라미터를 모델에 전달하는 방법이다. 여기에 포함된 하이퍼파라미터는 `n_estimators`, `max_depth`, `min_child_weight`, `learning_rate` 등이다. 이러한 하이퍼파라미터는 모델의 성능과 복잡성에 영향을 미친다.

* `xgb_model.fit(X_train, y_train)`:

   XGBoost 분류기를 훈련 데이터에 적합시키는 과정이다. `X_train`은 훈련 데이터의 특성을 포함하고, `y_train`은 훈련 데이터의 레이블(정답)을 포함시킨다. 이렇게 하면 모델이 훈련 데이터에서 패턴을 학습하고, 새로운 데이터에 대한 예측을 수행할 수 있게 된다.







```
param_grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [None, 10, 20, 30],
              'min_child_weight': [1, 2, 4],
              'learning_rate': [0.1, 0.01, 0.001]}
```

* `param_grid`는 그리드 검색(GridSearchCV)에서 탐색할 하이퍼파라미터의 값의 조합을 정의하는 딕셔너리이다. 여기서는 XGBoost 모델의 하이퍼파라미터 중 4개를 조절하고 있다.
* `n_estimators`: 부스트 트리의 개수를 의미한다. 더 많은 트리가 모델의 성능을 향상시킬 수 있지만, 너무 많은 트리는 오버피팅(과적합)을 일으킬 수 있다.
* `max_depth`: 각 트리의 최대 깊이를 의미한다. 더 깊은 트리는 더 복잡한 모델을 만들어 성능을 향상시킬 수 있지만, 오버피팅의 위험이 있다.
* `min_child_weight`: 트리에서 최소 가중치 합을 만족해야하는 자식 노드의 수. 이 값이 크면 트리는 더 적은 노드를 가지게 되어 모델의 복잡성이 줄어들어 오버피팅을 방지할 수 있다.
* `learning_rate`: 각 트리의 기여도를 조절하는 학습률. 작은 학습률은 더 많은 트리를 필요로 하지만, 일반적으로 더 나은 성능을 얻을 수 있다.

**더 좋은 결과를 얻기 위한 수정 방법 고안**

1. 하이퍼파라미터 범위 확장: `param_grid`의 각 하이퍼파라미터 범위를 확장하여 더 다양한 조합을 탐색할 수 있다. 예를 들어, `n_estimators`의 최대값을 300 또는 400으로 늘리거나, `learning_rate`에 0.05와 같은 다른 값을 추가한다.
2. 더 많은 하이퍼파라미터를 조절: XGBoost에는 다른 여러 하이퍼파라미터가 있다. 예를 들어, `subsample`과 `colsample_bytree`와 같은 일부 하이퍼파라미터를 추가하여 모델의 성능을 더 향상시킬 수 있다.
3. 무작위 검색 사용: 그리드 검색은 모든 가능한 조합을 탐색하므로 시간이 오래 걸릴 수 있다. 무작위 검색(RandomizedSearchCV)을 사용하여 주어진 시간 동안 무작위로 하이퍼파라미터 조합을 탐색하도록 할 수 있다. 

* 무작위 검색을 사용하려면 `RandomizedSearchCV`를 `GridSearchCV` 대신 사용하고, 각 하이퍼파라미터의 가능한 값 범위를 지정해야 한다. `Xgboost2`에서는 `RandomizedSearchCV`를 사용하여 수정해 볼 것이다.





작성중