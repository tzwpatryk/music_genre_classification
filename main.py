import os
import random
import streamlit as st
import matplotlib.pyplot as plt
from pydub import AudioSegment
from xgboost import XGBClassifier
import librosa
import librosa.display


def extract_features(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = chroma.mean()
    chroma_var = chroma.var()

    rms = librosa.feature.rms(y=y)
    rms_mean = rms.mean()
    rms_var = rms.var()

    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    cent_mean = cent.mean()
    cent_var = cent.var()

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spec_bw_mean = spec_bw.mean()
    spec_bw_var = spec_bw.var()

    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = rolloff.mean()
    rolloff_var = rolloff.var()

    zero = librosa.feature.zero_crossing_rate(y)
    zero_mean = zero.mean()
    zero_var = zero.var()

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]

    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    mfcc_mean = mfcc.mean()
    mfcc_var = mfcc.var()

    return [[chroma_mean, chroma_var, rms_mean, rms_var, cent_mean, cent_var, spec_bw_mean, spec_bw_var, rolloff_mean, rolloff_var, zero_mean, zero_var, tempo, mfcc_mean, mfcc_var]]


st.set_page_config(page_title='The Machine Learning App', layout='wide')

modelxgb = XGBClassifier()
modelxgb.load_model('modelxgb.json')

with st.header('1. Upload your song in wav or mp3 format:'):
    uploaded_file = st.file_uploader('Upload your input wav or mp3 file', type=['mp3', 'wav'])

songfile = 'hiphop00048.wav'
y, sr = librosa.load(songfile)
if st.button('Random song'):
    songfile = random.choice(['blues00048.wav', 'hiphop00048.wav', 'metal00099.wav'])
    y, sr = librosa.load(songfile)
    try:
        uploaded_file = None
    except NameError:
        pass

if uploaded_file is not None:
    if uploaded_file.name.endswith('.mp3'):
        sound = AudioSegment.from_mp3(uploaded_file)
        sound.export('uploaded.wav', format="wav")
        y, sr = librosa.load('uploaded.wav')
        st.audio('uploaded.wav')
        os.remove('uploaded.wav')
    else:
        y, sr = librosa.load(uploaded_file)
        st.audio(uploaded_file)
else:
    st.audio(songfile)

features = extract_features(y, sr)
y_pred = modelxgb.predict_proba(features)[0]

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

fig, ax = plt.subplots(figsize=(10, 4))
plt.bar(class_names, y_pred, color='maroon')
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#0e1117')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['top'].set_color('#0e1117')
ax.spines['right'].set_color('#0e1117')
ax.tick_params(colors='white', which='both')
st.pyplot(fig)

st.write("All sample songs are taken from GTZAN Dataset [Kaggle link](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)")