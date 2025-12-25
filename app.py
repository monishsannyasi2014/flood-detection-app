import streamlit as st
import pandas as pd
import numpy as np
import keras
import re
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Flood Detection AI", layout="centered")

# 2. Load the Model
# Cache the model so it doesn't reload every time you click a button
@st.cache_resource
def load_my_model():
    # Keras 3 automatically handles the .keras format
    model = keras.models.load_model("fine_tuned_flood_detection_model.keras")
    return model

model = load_my_model()

# 3. Helper Function to Clean Column Names
# This prevents the LightGBM/JSON error we fixed earlier
def sanitize_columns(df):
    df.columns = ["".join(re.findall(r'\w+', col)) for col in df.columns]
    return df

# 4. App Interface
st.title("🌊 Flood Detection System")
st.write("Upload an image or enter data to predict flood risk.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing for Keras 3 model
    # Note: Input shape is [batch, height, width, 3]
    img = image.resize((224, 224)) # Adjust size to match your model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if st.button("Predict Risk"):
        prediction = model.predict(img_array)
        st.write(f"### Prediction Result: {prediction[0]}")