import tensorflow as tf
import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

with open('model.json', 'r') as json_file:
    json_savedModel = json_file.read()

model_j = tf.keras.models.model_from_json(json_savedModel)
model_j.load_weights('model_weights.h5')
st.set_page_config(page_title='The Machine Learning App', layout='wide')

with st.subheader('1. Upload your song in wav format:'):
    uploaded_file = st.file_uploader('Upload your input wav file', type=['wav'])

if uploaded_file is not None:
    fig, ax = plt.subplots()
    y, sr = librosa.load(uploaded_file)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    plt.axis('off')
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=ax)
    plt.savefig('img.png')
try:
    test_image = tf.keras.utils.load_img('img.png', target_size=(432, 288))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
except FileNotFoundError:
    test_image = tf.keras.utils.load_img('images_original/blues/blues00000.png', target_size=(432, 288))
    test_image = tf.keras.utils.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)

prediction = model_j.predict(test_image)

class_names = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

prob_list = np.array(tf.nn.softmax(prediction))[0]

fig = plt.figure(figsize=(10,5))
plt.bar(class_names, prob_list, color='maroon')

st.pyplot(fig)



