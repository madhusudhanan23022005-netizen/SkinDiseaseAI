import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("skin_disease_model_v2.keras")

class_names = [
    "Acne_Mild",
    "Acne_Moderate",
    "Acne_Severe",
    "Eczema_Mild",
    "Eczema_Moderate",
    "Eczema_Severe",
    "Normal"
]

st.title("🩺 Skin Disease Detection AI")
st.write("Upload a skin image to get an AI prediction.")

uploaded_file = st.file_uploader(
    "Choose a skin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict Disease"):

        img = image.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        
        

        prediction = model.predict(img_array, verbose=0)

        predicted_index = np.argmax(prediction[0])
        disease = class_names[predicted_index]
        confidence = float(prediction[0][predicted_index]) * 100

        st.success(f"Predicted Disease: {disease}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.warning(
            "This AI result is for educational/research purposes only "
            "and is not a medical diagnosis."
        )