
# ML_jieun





## 0. 테스트음원



```
https://gongu.copyright.or.kr/gongu/main/main.do

DISCO
69_디스코.mp3
Luv.mp3
청춘계급.mp3

POP
LoveMe.mp3
CURIOS DAY.mp3
YOUR EYES.mp3
```







## 1. CNN



### CNN1



**코드 실행 결과**

```
이 노래의 예상되는 장르는 blues 입니다.
32/32 - 0s - loss: 3.4569 - accuracy: 0.5235 - 41ms/epoch - 1ms/step
테스트 정확도: 0.5235235095024109
```

으로 정확도는 `0.5235..`이지만 테스트파일의 실제 장르가 blues이므로 데이터의 장르 예측에는 성공했다.



### CNN2



CNN1에서 모델 아키텍처 변경으로 Conv2D 레이어의 필터 개수를 16에서 32, 32에서 64, 64에서 128로 늘리고, Dense 레이어의 개수를 1개에서 2개로 늘렸습니다. 또한, 두 번째 Dense 레이어 뒤에 Dropout 레이어를 추가하여 overfitting을 방지하도록 했다.



**코드 실행 결과**

```
이 노래의 예상되는 장르는 blues 입니다.
32/32 - 0s - loss: 3.3821 - accuracy: 0.6156 - 53ms/epoch - 2ms/step
테스트 정확도: 0.6156156063079834
```

으로 정확도는 `0.6156..`이지만 테스트파일의 실제 장르가 blues이므로 데이터의 장르 예측에는 성공했다.







## 2. KNN



### knn1



**코드 실행 결과**

```
예측한 음악 장르: metal
정확도: 64.50%
```

으로 정확도는 `64.50%`지만 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.





### knn2



* knn1의 코드에서 교차 검증을 사용하여 정확도를 높이고, 그리드 검색을 사용하여 최적의 k값을 찾아 수정해보았다.

**코드 실행 결과**

```
최적 k 값: 5
예측한 음악 장르: metal
정확도: 64.50%
교차 검증된 정확도: 66.75%
```

으로 정확도는 `64.50%`, 교차 검증된 정확도는 `66.75%` 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.





## 3. RandomForestClassifier



* `knn2`의 코드에서 모델을 변경하여, `RandomForestClassifier`로 모델을 생성하고, 그리드 검색을 사용하여 최적의 하이퍼파라미터를 찾아 정확도를 높이는 시도를 해 보았다.



**코드 실행 결과**

```
최적의 하이퍼파라미터: {'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 76.00%
교차 검증된 정확도: 78.62%
```

으로 정확도는 `76.00%`, 교차 검증된 정확도는 `78.62%` 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.







**features_3_sec.csv 데이터로 변경 후 코드 실행 결과**

```
최적의 하이퍼파라미터: {'max_depth': 20, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 88.14%
교차 검증된 정확도: 85.82%
```

으로 정확도는 `88.14%`, 교차 검증된 정확도는 `85.82%` 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.











## 4. xgboost



* `RandomForestClassifier`를 사용한 모델링에서 `XGBoost`를 사용하여 바꾸어보았다.
* `XGBoost`는 분류 및 회귀 분석에 모두 사용이 가능하며, 고차원의 특성과 대량의 데이터에 대해 높은 성능을 발휘하여 높은 정확도와 빠른 실행 속도를 보장한다.

* `XGBoost`는 `Gradient Boosting` 알고리즘을 기반으로 하며, 여러 개의 결정 트리를 사용하여 모델을 구성하므로 `RandomForestClassifier`보다 성능이 우수할 것이란 예측을 했다.
* 또한 ` XGBoost`는 하이퍼파라미터 튜닝을 통해 모델의 성능을 더욱 개선할 수 있다. 
  `GridSearchCV`와 같은 교차 검증 기법을 사용하여 최적의 하이퍼파라미터를 자동으로 찾을 수 있다.



###  xgboost1



1. `extract_features` 함수를 정의하여 음악 파일에서 MFCC, Chroma, Mel, Spectral Contrast, Tonnetz와 같은 오디오 특성을 추출한다.
2. 데이터셋을 불러와 문자열 클래스 레이블을 숫자형으로 변환한다.
3. 특성과 레이블을 분리하고, 데이터를 학습용과 테스트용으로 분리한다. (예: 80% 학습, 20% 테스트)
4. 특성 스케일링을 수행한다 (StandardScaler 사용).
5. XGBoost 분류기를 생성하고 GridSearchCV를 사용하여 하이퍼파라미터 튜닝을 수행한다.
6. 최적의 하이퍼파라미터를 사용하여 XGBoost 분류기를 학습시킨다.
7. 테스트 파일에서 특성을 추출하고, 스케일링된 특성을 사용하여 장르를 예측한다.
8. 테스트 세트에서 정확도를 평가하고 교차 검증을 사용하여 정확도를 평가한다.



**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.1, 'max_depth': None, 'min_child_weight': 2, 'n_estimators': 200}
예측한 음악 장르: classical
정확도: 75.50%
교차 검증된 정확도: 77.25%
```

으로 정확도는 `75.50%`, 교차 검증된 정확도는 `77.25%` 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.





###  xgboost2



1. 필요한 라이브러리를 추가 임포트:
   - `RandomizedSearchCV`: 하이퍼파라미터 튜닝에 사용된다.
   - `randint`, `uniform`: 하이퍼파라미터 튜닝에서 무작위 값을 생성하는 데 사용된다.
2. `extract_features` 함수를수정:
   - `n_features` 매개변수를 추가하여 함수 호출시 원하는 특성 개수를 지정할 수 있도록 했다.
   - `feature_vector` 변수를 생성하여 여러 특성을 하나의 배열로 결합하고, 원하는 개수의 특성으로 자른다.
3. 데이터 분할 과정에서 `random_state`를 제거.
4. XGBoost 분류기 생성 시 `random_state`를 제거.
5. GridSearchCV 대신 RandomizedSearchCV를 사용하여 하이퍼파라미터 튜닝을 수행했다:
   - 무작위 값이 있는 `param_dist` 딕셔너리를 정의.
   - `random_search`를 사용하여 하이퍼파라미터 탐색을 수행하고 최적의 하이퍼파라미터를 출력.
6. 수정된 `extract_features` 함수를 사용하여 테스트 파일의 특성을 추출.





**코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.17940931103542268, 'max_depth': 10, 'min_child_weight': 1, 'n_estimators': 295}
예측한 음악 장르: classical
정확도: 76.50%
교차 검증된 정확도: 76.88%
```

으로 정확도는 `76.50%`, 교차 검증된 정확도는 `76.88%` 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.





**features_3_sec.csv 데이터로 변경 후 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.09796599427179664, 'max_depth': None, 'min_child_weight': 3, 'n_estimators': 243}
예측한 음악 장르: classical
정확도: 90.14%
교차 검증된 정확도: 88.39%
```

으로 정확도는 `90.14%`, 교차 검증된 정확도는 `88.39%` 테스트파일의 실제 장르는 blues로 데이터의 장르 예측에는 실패했다.







###  xgboost3







학습된 결과를 `Confusion Matrix plot`으로 만들고 .png 파일로 저장합니다.

처음엔 `plot_confusion_matrix` 함수를 사용했지만, 오류가 반복적으로 발생하여. 직접 `Confusion Matrix`를 그리는 방법으로 대체했습니다.







**데이터 셋을 test_dataset.csv로 설정한 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.11712624707999042, 'max_depth': 18, 'min_child_weight': 1, 'n_estimators': 299}
예측한 음악 장르: classical
정확도: 94.67%
교차 검증된 정확도: 91.17%
```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/confusion_matrix_test_dataset.png)





**데이터 셋을 train_dataset.csv로 설정한 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.08799815109814886, 'max_depth': None, 'min_child_weight': 1, 'n_estimators': 283}
예측한 음악 장르: classical
정확도: 90.56%
교차 검증된 정확도: 89.90%
```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/confusion_matrix_train_dataset.png)









###  xgboost4



음원 파일을 3초씩 끊어서 0초부터 30초까지 예측한 음악 장르를 볼 수 있도록 코드를 수정했습니다.



**데이터 셋을 test_dataset.csv로 설정한 코드 실행 결과**

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





###  xgboost5



테스트 음원의 길이가 너무 짧아서 오류를 발생시켜 코드에 예외 처리를 추가했다.





**disco1.wav 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.14195598843395105, 'max_depth': 13, 'min_child_weight': 1, 'n_estimators': 272}
0-3초
예측한 음악 장르: classical

3-6초
예측한 음악 장르: classical

6-9초
예측한 음악 장르: classical

9-12초
예측한 음악 장르: pop

12-15초
예측한 음악 장르: pop

15-18초
예측한 음악 장르: jazz

18-21초
예측한 음악 장르: jazz

C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\librosa\core\spectrum.py:256: UserWarning: n_fft=1024 is too large for input 
signal of length=739
  warnings.warn(
21-24초
예측한 음악 장르: classical

오류 발생 (offset: 24, duration: 3): module 'soundfile' has no attribute 'SoundFileRuntimeError'
24-27초 예측 실패 (오류)

오류 발생 (offset: 27, duration: 3): module 'soundfile' has no attribute 'SoundFileRuntimeError'
27-30초 예측 실패 (오류)

정확도: 92.33%
교차 검증된 정확도: 92.00%
```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/Xgboost5_disco1.png)





**pop1.wav 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.19535337721066814, 'max_depth': None, 'min_child_weight': 2, 'n_estimators': 284}
0-3초
예측한 음악 장르: pop

3-6초
예측한 음악 장르: pop

6-9초
예측한 음악 장르: pop

9-12초
예측한 음악 장르: pop

12-15초
예측한 음악 장르: pop

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

정확도: 93.50%
교차 검증된 정확도: 92.12%
```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/Xgboost5_pop1.png)









###  xgboost6



XGBoost 모델을 CPU만으로 학습을 진행하니, 시간이 너무 오래걸려 새로운 방법을 찾아 GPU를 사용하여 학습시켜본다.





**disco2.wav 코드 실행 결과**

```
GPU가 있습니다.
최적의 하이퍼파라미터: {'learning_rate': 0.17004834788079481, 'max_depth': 26, 'min_child_weight': 4, 'n_estimators': 260}
0-3초
예측한 음악 장르: classical

3-6초
예측한 음악 장르: classical

6-9초
예측한 음악 장르: classical

9-12초
예측한 음악 장르: classical

12-15초
예측한 음악 장르: classical

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

정확도: 93.17%
교차 검증된 정확도: 91.33%

```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/Xgboost6_disco2.png)

