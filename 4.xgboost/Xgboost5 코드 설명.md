
# Xgboost5 코드 설명



* `Xgboost4`에서 변경 한 부분만 설명하였습니다.



**새로운 테스트 음원(disco1.wav)을 넣었더니** 



C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\librosa\core\spectrum.py:256: UserWarning: n_fft=1024 is too large for input 
signal of length=739
  warnings.warn(
21-24초
예측한 음악 장르: classical

Traceback (most recent call last):
  File "C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\librosa\core\audio.py", line 176, in load
    y, sr_native = __soundfile_load(path, offset, duration, dtype)
  File "C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\librosa\core\audio.py", line 215, in __soundfile_load
    sf_desc.seek(int(offset * sr_native))
  File "C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\soundfile.py", line 870, in seek
    _error_check(self._errorcode)
  File "C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\soundfile.py", line 1455, in _error_check
    raise RuntimeError(prefix + _ffi.string(err_str).decode('utf-8', 'replace'))
RuntimeError: Internal psf_fseek() failed.

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "c:\Users\JIEUN\Desktop\ML_jieun\4.xgboost\230417Xgboost4.py", line 106, in <module>
    predict_genre_segments(test_file_path, 0, 30, 3, 3,
  File "c:\Users\JIEUN\Desktop\ML_jieun\4.xgboost\230417Xgboost4.py", line 95, in predict_genre_segments
    test_file_features = extract_features_with_offset(
  File "c:\Users\JIEUN\Desktop\ML_jieun\4.xgboost\230417Xgboost4.py", line 35, in extract_features_with_offset
    audio_data, sample_rate = librosa.load(
  File "C:\Users\JIEUN\AppData\Local\Programs\Python\Python39\lib\site-packages\librosa\core\audio.py", line 178, in load
    except sf.SoundFileRuntimeError as exc:
AttributeError: module 'soundfile' has no attribute 'SoundFileRuntimeError'



**오류가 나온다. 수정해보자.**



**이 오류의 이유로는 2가지가 추측된다.**

1. 테스트 음원의 길이가 너무 짧아서 n_fft 값이 입력 신호의 길이보다 큰 경우다. 이 경우에는 n_fft 값을 줄이거나, 입력 신호의 길이를 늘릴 수 있다.
2. 테스트 음원의 길이가 지정한 구간의 합보다 짧은 경우다. 이 경우에는 구간을 줄이거나, 음원의 길이를 늘릴 수 있다.





코드를 수정하여 예외 처리를 추가하고, 예외 발생 시 메시지를 출력하도록 변경해본다.

```python
def extract_features_with_offset(file_path, n_features, offset, duration):
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=None, mono=True, offset=offset, duration=duration)
        # 나머지 부분은 기존 extract_features 함수와 동일
    except Exception as e:  # 예외 처리 추가
        print(f"오류 발생 (offset: {offset}, duration: {duration}): {e}")
        return None  # 오류가 발생하면 None을 반환

def predict_genre_segments(file_path, start, end, step, duration, model, scaler, le, n_features):
    for i in range(start, end, step):
        test_file_features = extract_features_with_offset(file_path, n_features, i, duration)
        if test_file_features is not None:  # 예외 처리 추가: test_file_features가 None이 아닌 경우에만 실행
            test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))
            predicted_genre = model.predict(test_file_features_scaled)
            predicted_genre_str = le.inverse_transform(predicted_genre)
            print(f"{i}-{i+duration}초")
            print(f"예측한 음악 장르: {predicted_genre_str[0]}\n")
        else:
            print(f"{i}-{i+duration}초 예측 실패 (오류)\n")  # 예외 처리 추가: 오류 메시지 출력
```









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

![](https://github.com/ZBDS11ML3/ML_jieun/blob/main/0.Confusion_matrix/Xgboost4_disco1.png)



