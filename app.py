import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5", compile=False)

model = load_model()

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

st.title("Fashion Product Classification (CNN + AI)")

st.write("Upload an image and the model will predict the fashion category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize((28, 28))

    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    st.write("### Prediction:", class_names[predicted_class])
    st.write("Confidence:", float(np.max(prediction)))
