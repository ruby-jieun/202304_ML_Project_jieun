import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess


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
    feature_vector = np.hstack(
        [mfccs, chroma, mel, contrast, tonnetz])
    return feature_vector[:n_features]


def extract_features_with_offset(file_path, n_features, offset, duration):
    try:
        audio_data, sample_rate = librosa.load(
            file_path, sr=None, mono=True, offset=offset, duration=duration)
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
    except Exception as e:
        print(f"오류 발생 (offset: {offset}, duration: {duration}): {e}")
        return None


data = pd.read_csv("4.xgboost/Data/test_dataset.csv")

le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])

X = data.drop(['filename', 'label'], axis=1).values
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

n_features = X_train.shape[1]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost 모델 생성 시 GPU 사용 설정
xgb_model = xgb.XGBClassifier(
    tree_method='gpu_hist', gpu_id=0) if check_gpu() else xgb.XGBClassifier()

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

xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train)


test_file_path = "4.xgboost/Data/genres_test/disco/disco2.wav"


def predict_genre_segments(file_path, start, end, step, duration, model, scaler, le, n_features):
    for i in range(start, end, step):
        test_file_features = extract_features_with_offset(
            file_path, n_features, i, duration)
        if test_file_features is not None:
            test_file_features_scaled = scaler.transform(
                test_file_features.reshape(1, -1))
            predicted_genre = model.predict(test_file_features_scaled)
            predicted_genre_str = le.inverse_transform(predicted_genre)
            print(f"{i}-{i+duration}초")
            print(f"예측한 음악 장르: {predicted_genre_str[0]}\n")
        else:
            print(f"{i}-{i+duration}초 예측 실패 (오류)\n")


predict_genre_segments(test_file_path, 0, 30, 3, 3,
                       xgb_model, scaler, le, n_features)

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy * 100:.2f}%")

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f"교차 검증된 정확도: {np.mean(cv_scores) * 100:.2f}%")

genre_classes = le.inverse_transform(np.unique(y))


def save_confusion_matrix_plot(model, X_test, y_test, class_names, file_name):
    plt.figure(figsize=(10, 10))
    cm = confusion_matrix(y_test, model.predict(X_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, annot=True, cmap=plt.cm.Blues,
                     xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(file_name)
    plt.show()


save_confusion_matrix_plot(xgb_model, X_test, y_test,
                           genre_classes, '0.Confusion_matrix/Xgboost6_disco3.png')
