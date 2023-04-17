import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix  # 수정된 부분
from sklearn.preprocessing import LabelEncoder
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns  # 추가된 부분


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


data = pd.read_csv("4.xgboost/Data/train_dataset.csv")

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

xgb_model = xgb.XGBClassifier()

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

test_file_path = "4.xgboost/Data/genres_original/blues/blues.00000.wav"
test_file_features = extract_features(test_file_path, n_features)

test_file_features_scaled = scaler.transform(test_file_features.reshape(1, -1))

predicted_genre = xgb_model.predict(test_file_features_scaled)

predicted_genre_str = le.inverse_transform(predicted_genre)
print(f"예측한 음악 장르: {predicted_genre_str[0]}")

y_pred = xgb_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"정확도: {accuracy * 100:.2f}%")

cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5)
print(f"교차 검증된 정확도: {np.mean(cv_scores) * 100:.2f}%")

genre_classes = le.inverse_transform(np.unique(y))  # 추가된 부분

# 추가된 함수: Confusion Matrix plot 저장


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


# 학습된 모델로부터 Confusion Matrix plot 저장
save_confusion_matrix_plot(xgb_model, X_test, y_test,
                           genre_classes, '0.Confusion_matrix/confusion_matrix.png')
