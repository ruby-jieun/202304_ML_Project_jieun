import os
import numpy as np
import tensorflow as tf
import librosa
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import soundfile as sf

# 데이터 경로
data_dir = '1.CNN/Data/genres_original'

# 데이터 전처리 함수


def preprocess_data(data_dir):
    X = []
    y = []
    labels = os.listdir(data_dir)
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        for file_name in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file_name)
            # 음성 파일 불러오기
            try:
                signal, sr = sf.read(file_path)
            except (RuntimeError, FileNotFoundError, EOFError, ZeroDivisionError) as e:
                # print(f'불러올 수 없는 파일 : {file_path}. {e}')
                continue
            # 음성 파일에서 특징 추출
            mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            # 데이터와 레이블 추가
            X.append(mfccs_processed)
            y.append(label)
    # 레이블 인코딩
    le = LabelEncoder()
    y = le.fit_transform(y)
    # 레이블을 one-hot encoding으로 변환
    y = to_categorical(y)
    # 데이터 정규화
    X = np.array(X)
    X = (X - np.mean(X)) / np.std(X)
    X = X[..., np.newaxis]
    return X, y, le.classes_


# 데이터 전처리
X, y, class_names = preprocess_data(data_dir)

# 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((X.shape[1], 1, 1), input_shape=X.shape[1:]),
    tf.keras.layers.Conv2D(16, 3, padding='same',
                           activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.summary()

# 모델 컴파일
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
history = model.fit(X, y, epochs=50, validation_split=0.2)

# 모델 평가
# test_X, test_y, _ = preprocess_data(data_dir)
# test_loss, test_acc = model.evaluate(test_X, test_y, verbose=2)
# print('테스트 정확도:', test_acc)


# 테스트 음원 예측
def predict_genre(file_path):
    # 음성 파일 불러오기
    try:
        signal, sr = sf.read(file_path)
    except (RuntimeError, FileNotFoundError, EOFError, ZeroDivisionError) as e:
        # print(f'불러올 수 없는 파일 : {file_path}. {e}')
        return
    # 음성 파일에서 특징 추출
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    # 모델에 입력하기 위해 데이터 전처리
    X, _, _ = preprocess_data(data_dir)
    X = (mfccs_processed - np.mean(X)) / np.std(X)
    X = X[np.newaxis, ..., np.newaxis]
    # 예측
    prediction = model.predict(X)
    # 예측 결과 출력
    predicted_genre = class_names[np.argmax(prediction)]
    print(f'이 노래의 예상되는 장르는 {predicted_genre} 입니다.')
    return predicted_genre


# 테스트 음원 경로
test_file_path = '1.CNN/Data/genres_original/blues/blues.00000.wav'

# 테스트 음원 예측 및 정확도 출력
predicted_genre = predict_genre(test_file_path)
if predicted_genre:
    test_X, test_y, _ = preprocess_data(data_dir)
    test_loss, test_acc = model.evaluate(test_X, test_y, verbose=2)
    print('테스트 정확도:', test_acc)


'Data/believe-me-143530-cut-30sec.wav'
