
# Xgboost6 코드 설명



* `Xgboost5`에서 변경 한 부분만 설명하였습니다.



**XGBoost 모델을 CPU만으로 학습을 진행하니, 너무 오래걸렸다. **
**새로운 방법을 찾아 GPU를 사용하여 학습시켜보려한다.** 



XGBoost 모델을 GPU를 사용하여 학습하고 예측을 수행하면 다음과 같은 장점이 있다.

1. **빠른 학습과 예측**: GPU는 대규모 행렬 연산을 빠르게 처리할 수 있기 때문에, CPU에 비해 XGBoost 모델의 학습과 예측 속도를 크게 향상시킬 수 있다.
2. **대용량 데이터 처리**: 대용량 데이터를 처리할 때, GPU를 사용하면 처리 시간을 대폭 줄일 수 있다. 이는 XGBoost 모델의 학습과 예측 뿐 아니라, 대용량 데이터를 전처리하는 과정에서도 유용하다.
3. **모델 성능 개선**: GPU를 사용하면 모델 학습에 필요한 시간을 단축시켜 더 많은 하이퍼파라미터를 탐색할 수 있다. 이는 모델의 성능을 더욱 개선시킬 수 있는데, 더 좋은 성능을 얻기 위해서는 다양한 하이퍼파라미터를 시도해 봐야하기 때문이다.
4. **비용 절감**: GPU를 사용하여 모델 학습을 더욱 빠르게 수행할 수 있으므로, 모델 학습에 필요한 인프라 비용을 줄일 수 있다. 이는 대규모 데이터를 처리하는 기업이나 연구자들에게 큰 이점이 된다.



GPU 사용 가능 여부를 확인하기 위해 import torch와, import tensorflow as tf를 이용해봤지만 제 PC에서 인식하지 못해 새로운 방법을 찾아보았습니다.



NVIDIA System Management Interface(`nvidia-smi`)를 사용해보았습니다.

```python
import subprocessxxxxxxxxxx import subprocessimport torch
```



시스템에 GPU가 있는지 확인한다.

```python
def check_gpu():
    try:
        output = subprocess.check_output("nvidia-smi", shell=True)
        return "No devices were found" not in output.decode("utf-8")
    except Exception as e:
        print(f"오류 발생: {e}")
        return False

if check_gpu():
    print("GPU가 있습니다.")
else:
    print("GPU가 없습니다.")
```



XGBoost 모델 생성 시 GPU를 사용하도록 설정한다.

```python
xgb_model = xgb.XGBClassifier(tree_method='gpu_hist', gpu_id=0) if check_gpu() else xgb.XGBClassifier()
```







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





**disco3.wav 코드 실행 결과**

```
최적의 하이퍼파라미터: {'learning_rate': 0.07870781512814429, 'max_depth': None, 'min_child_weight': 2, 'n_estimators': 268}
0-3초
예측한 음악 장르: classical

3-6초
예측한 음악 장르: classical

6-9초
예측한 음악 장르: reggae

9-12초
예측한 음악 장르: reggae

12-15초
예측한 음악 장르: classical

15-18초
예측한 음악 장르: classical

18-21초
예측한 음악 장르: classical

21-24초
예측한 음악 장르: classical

24-27초
예측한 음악 장르: reggae

27-30초
예측한 음악 장르: pop

정확도: 93.50%
교차 검증된 정확도: 91.92%
```

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/Xgboost6_disco3.png)