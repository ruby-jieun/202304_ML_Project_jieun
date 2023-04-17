# 필요한 라이브러리를 임포트합니다.
import os
import librosa
import pandas as pd
import numpy as np

# WAV 파일에서 특성을 추출하는 함수를 정의합니다.


def extract_features(file):
    # WAV 파일을 로드하고, 오디오 신호(y)와 샘플링 레이트(sr)를 얻습니다.
    y, sr = librosa.load(file, mono=True)

    # 다양한 오디오 특성을 추출합니다.
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    harmony = librosa.effects.harmonic(y)
    perceptr = librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    # 추출한 특성의 평균과 분산을 계산합니다.
    feature_data = {
        'chroma_stft_mean': np.mean(chroma_stft),
        'chroma_stft_var': np.var(chroma_stft),
        'rms_mean': np.mean(rms),
        'rms_var': np.var(rms),
        'spectral_centroid_mean': np.mean(spec_cent),
        'spectral_centroid_var': np.var(spec_cent),
        'spectral_bandwidth_mean': np.mean(spec_bw),
        'spectral_bandwidth_var': np.var(spec_bw),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_var': np.var(rolloff),
        'zero_crossing_rate_mean': np.mean(zcr),
        'zero_crossing_rate_var': np.var(zcr),
        'harmony_mean': np.mean(harmony),
        'harmony_var': np.var(harmony),
        'perceptr_mean': np.mean(perceptr),
        'perceptr_var': np.var(perceptr),
        'tempo': tempo
    }

    # MFCC 특성의 평균과 분산을 계산하고, feature_data 딕셔너리에 추가합니다.
    for i, mfcc_value in enumerate(mfcc):
        feature_data[f'mfcc{i+1}_mean'] = np.mean(mfcc_value)
        feature_data[f'mfcc{i+1}_var'] = np.var(mfcc_value)

    # 추출한 특성 데이터를 반환합니다.
    return feature_data


# WAV 파일 경로를 지정합니다.
wav_file_path = '4.xgboost/Data/genres_test/disco/disco1.wav'

# WAV 파일에서 특성을 추출합니다.
features = extract_features(wav_file_path)

# 추출한 특성을 데이터프레임으로 변환합니다.
df = pd.DataFrame(features, index=[0])

# 데이터프레임을 CSV 파일로 저장합니다.
df.to_csv('4.xgboost/Data/genres_test/disco/disco1.csv', index=False)
