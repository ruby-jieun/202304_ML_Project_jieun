
# Xgboost4 코드 설명



* `Xgboost3`에서 변경 한 부분만 설명하였습니다.



음원 파일을 3초씩 끊어서 0초부터 30초까지 예측한 음악 장르를 볼 수 있도록 코드를 수정했습니다.





1. `extract_features_with_offset` 함수를 추가했습니다. 이 함수는 `offset`과 `duration` 인자를 추가로 받아 음원의 특정 구간만 특징을 추출하도록 변경되었습니다.

```python
def extract_features_with_offset(file_path, n_features, offset, duration):
    audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True, offset=offset, duration=duration)
    # 나머지 부분은 기존 extract_features 함수와 동일
```



2. `predict_genre_segments` 함수를 추가했습니다. 이 함수는 주어진 경로의 음악 파일에서 구간별로 음악 장르를 예측합니다.

```python
def predict_genre_segments(file_path, start, end, step, duration, model, scaler, le, n_features):
    for i in range(start, end, step):
        test_file_features = extract_features_with_offset(file_path, n_features, i, duration)
        test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))
        predicted_genre = model.predict(test_file_features_scaled)
        predicted_genre_str = le.inverse_transform(predicted_genre)
        print(f"{i}-{i+duration}초")
        print(f"예측한 음악 장르: {predicted_genre_str[0]}\n")
```



3. 기존 코드의 예측 부분을 수정하여 `predict_genre_segments` 함수를 호출하도록 변경했습니다.

```python
# 수정된 부분: 0초부터 30초까지 3초 간격으로 음악 장르 예측
predict_genre_segments(test_file_path, 0, 30, 3, 3, xgb_model, scaler, le, n_features)
```



위의 수정된 부분은 기존 코드에 구간별 음악 장르 예측 기능을 추가하기 위해 만들어졌습니다. `extract_features_with_offset` 함수는 특정 구간의 특징을 추출하고, `predict_genre_segments` 함수는 해당 구간별 음악 장르를 예측하는 기능을 제공합니다. 마지막으로, 기존 코드의 예측 부분을 수정하여 `predict_genre_segments` 함수를 호출함으로써 0초부터 30초까지 3초 간격으로 음악 장르를 예측하도록 변경했습니다.





**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.11803817668718833, 'max_depth': None, 'min_child_weight': 3, 'n_estimators': 201}
0-3초
예측한 음악 장르: classical

3-6초
예측한 음악 장르: classical

6-9초
예측한 음악 장르: classical

9-12초
예측한 음악 장르: classical

12-15초
예측한 음악 장르: reggae

15-18초
예측한 음악 장르: classical

18-21초
예측한 음악 장르: classical

21-24초
예측한 음악 장르: classical

24-27초
예측한 음악 장르: classical

27-30초
예측한 음악 장르: classical

정확도: 92.17%
교차 검증된 정확도: 92.42%
```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/Xgboost4.png)





