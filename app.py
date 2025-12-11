import os
import keras
from keras.models import load_model
import tensorflow as tf
import streamlit as st
import pickle
import numpy as np

st.header("Flower Type Prediction")
model_file = 'flower_names.bin'
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
img_size = 180

with open('flower_names.bin', 'rb') as f_in:
    model = pickle.load(f_in)


def classify_flower(image_path):
    input_img = tf.keras.utils.load_img(image_path, target_size=(img_size, img_size))
    input_image_array = tf.keras.utils.img_to_array(input_img)
    input_image_exp_dim = tf.expand_dims(input_image_array, axis=0)
    predictions = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(predictions[0])
    output = ' The Imnage is of type: ' + flower_names[np.argmax(result)] + ' with a score of ' + str(max(result)*100)
    return output

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, width=200)

    st.markdown(classify_flower(os.path.join("tempDir", uploaded_file.name)))