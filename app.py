import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model(r"G:\pneumonia_detection\pneumonia_model.keras")


# Function to predict pneumonia
def predict(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0 \\

    prediction = model.predict(img_array)
    return "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

# Streamlit UI
st.title("Pneumonia Detection System")
st.write("Upload a Chest X-ray Image to Predict Pneumonia.")

uploaded_file = st.file_uploader("Choose an X-ray Image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    # Save uploaded image
    with open("uploaded.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = predict("uploaded.jpg")
    st.write(f"Prediction: *{result}*")