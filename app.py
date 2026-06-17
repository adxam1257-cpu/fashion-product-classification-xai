import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

@st.cache_resource
def load_model():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28, 28)),
        tf.keras.layers.Reshape((28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    weights_path = os.path.join(os.path.dirname(__file__), "model_weights.h5")
    model.load_weights(weights_path)
    return model

model = load_model()

st.title("Fashion Product Classification (CNN + XAI)")
st.write("Upload an image and the model will predict the fashion category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(image, caption="Uploaded Image", width=150)
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    st.success(f"Prediction: **{predicted_class}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
