import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("MNIST Digit Classifier")
model = tf.keras.models.load_model('mnist_model.h5')

uploaded_file = st.file_uploader("Upload an image of a digit (28x28 grayscale)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    st.image(img, caption="Uploaded Digit", width=150)
    st.write("Prediction:", np.argmax(prediction)) 