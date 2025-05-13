import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import time

# Load model (adjust path as necessary)
model = tf.keras.models.load_model(r"C:\Users\SATYA\final_lung_disease_model.keras")


# Class labels (adjust based on your dataset)
class_names = ['Bacterial Pneumonia', 'Corona Virus Disease', 'Normal', 'Tuberculosis', 'Viral Pneumonia']

# Streamlit UI
st.set_page_config(page_title="Lung Disease Detector", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .title {
        font-size:40px;
        color: #0e5ef9;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="title">🫁 Lung Disease Detection Using AI</div>', unsafe_allow_html=True)

st.write("Upload a chest X-ray image to predict the type of lung disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing the image..."):
            time.sleep(1)  # For animation
            # Preprocess the image
            img = image.resize((224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            # Predict
            predictions = model.predict(img_array)
            class_index = np.argmax(predictions)
            confidence = np.max(predictions) * 100

            st.success("✅ Prediction Complete!")
            st.markdown(f"### 🩺 Detected: **{class_names[class_index]}**")
            st.progress(min(int(confidence), 100))
            st.write(f"Confidence: `{confidence:.2f}%`")

            # Optional: Display class probabilities
            st.subheader("Class Probabilities:")
            for i, prob in enumerate(predictions[0]):
                st.write(f"**{class_names[i]}**: {prob * 100:.2f}%")

else:
    st.info("Please upload an image to proceed.")

# Footer animation
st.markdown(
    """
    <hr>
    <p style="text-align: center;">Built with ❤️ using Streamlit & TensorFlow</p>
    """,
    unsafe_allow_html=True
)
