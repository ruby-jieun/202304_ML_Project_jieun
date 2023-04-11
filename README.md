# ML_jieun



1. **데이터 전처리 함수** `preprocess_data(data_dir)` : 주어진 디렉토리 `data_dir`에 있는 음악 파일들을 읽어와서 MFCC(Mel-frequency cepstral coefficients) 특징을 추출합니다. 이를 통해 음악 파일을 숫자 데이터로 변환하고, 이에 대한 레이블을 할당합니다. 그 후, 레이블을 one-hot 인코딩으로 변환하고, 데이터를 정규화합니다. 이렇게 변환된 데이터와 레이블을 반환합니다.
2. **모델 구성** : `tf.keras.Sequential()`을 사용하여 CNN 모델을 구성합니다. 모델의 입력은 `Reshape` 레이어를 통해 2D 배열로 변환된 데이터입니다. 이후 3개의 Conv2D 레이어와 MaxPooling2D 레이어가 번갈아가며 나타납니다. 이를 통해 모델은 공간적인 정보를 추출할 수 있습니다. 그리고 `Flatten()` 레이어를 거쳐서 1D 배열로 펼쳐지며, `Dense()` 레이어를 통해 마지막으로 softmax 활성화 함수를 사용하여 10개의 클래스를 구분합니다.
3. **모델 컴파일** : `compile()` 메서드를 사용하여 모델을 컴파일합니다. `categorical_crossentropy` 손실 함수와 `adam` 옵티마이저를 사용하며, `accuracy`를 모니터링합니다.
4. **모델 학습** : `fit()` 메서드를 사용하여 모델을 학습시킵니다. `X`와 `y` 데이터를 사용하며, `validation_split`을 사용하여 검증 데이터셋을 만듭니다. 학습된 모델은 `history` 객체에 저장됩니다.
5. **모델 평가** : `evaluate()` 메서드를 사용하여 학습된 모델의 정확도를 평가합니다.
6. **음원 예측 함수** `predict_genre(file_path)` : 모델을 사용하여 음원의 장르를 예측합니다. 입력으로 들어온 파일을 불러와서 MFCC 특징을 추출하고, 데이터 전처리 후, 모델을 사용하여 예측을 수행합니다. 이를 통해 예측된 장르를 출력합니다.
7. **테스트 음원 예측** : `predict_genre()` 함수를 사용하여 장르를 예측합니다. 그리고 `evaluate()` 메서드를 사용하여 모델의 정확도를 출력합니다.
