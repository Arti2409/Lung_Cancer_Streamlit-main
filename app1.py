
import streamlit as st
import joblib
import os
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu

# --------------------------------------------------
# PAGE CONFIG (ONLY ONCE)
# --------------------------------------------------
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")
st.title("🫁 Lung Cancer Detection System")

# --------------------------------------------------
# LOAD ML MODEL (SAFE)
# --------------------------------------------------
MODEL_PATH = "models/final_model.sav"

if not os.path.exists(MODEL_PATH):
    st.error("❌ ML model not found")
    st.stop()

@st.cache_resource
def load_ml_model():
    return joblib.load(MODEL_PATH)

model = load_ml_model()
st.success("✅ ML model loaded")

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    selection = option_menu(
        "Navigation",
        [
            "Introduction",
            "About the Dataset",
            "Lung Cancer Prediction",
            "CNN Based Disease Prediction"
        ],
        icons=["activity", "bar-chart", "person", "image"],
        default_index=0
    )


if selection == "Lung Cancer Prediction":

    st.header("🔍 Lung Cancer Prediction (ML Model)")

    col1, col2, col3 = st.columns(3)

    with col1:
        Age = st.number_input("Age", 1, 100, 30)
        Gender = st.number_input("Gender (0=Female, 1=Male)", 0, 1)
        AirPollution = st.number_input("Air Pollution", 0, 10)

    with col2:
        AlcoholUse = st.number_input("Alcohol Use", 0, 10)
        BalancedDiet = st.number_input("Balanced Diet", 0, 10)
        Obesity = st.number_input("Obesity", 0, 10)

    with col3:
        Smoking = st.number_input("Smoking", 0, 10)
        PassiveSmoker = st.number_input("Passive Smoker", 0, 10)
        Fatigue = st.number_input("Fatigue", 0, 10)

    WeightLoss = st.number_input("Weight Loss", 0, 10)
    ShortnessofBreath = st.number_input("Shortness of Breath", 0, 10)
    Wheezing = st.number_input("Wheezing", 0, 10)
    SwallowingDifficulty = st.number_input("Swallowing Difficulty", 0, 10)
    ClubbingofFingerNails = st.number_input("Clubbing of Finger Nails", 0, 10)
    FrequentCold = st.number_input("Frequent Cold", 0, 10)
    DryCough = st.number_input("Dry Cough", 0, 10)
    Snoring = st.number_input("Snoring", 0, 10)

    if st.button("Predict Lung Cancer Risk"):
        features = np.array([[
            Age, Gender, AirPollution, AlcoholUse, BalancedDiet, Obesity,
            Smoking, PassiveSmoker, Fatigue, WeightLoss,
            ShortnessofBreath, Wheezing, SwallowingDifficulty,
            ClubbingofFingerNails, FrequentCold, DryCough, Snoring
        ]])

        prediction = model.predict(features)[0]

        if prediction == "High":
            st.error("🔴 High Risk of Lung Cancer")
        elif prediction == "Medium":
            st.warning("🟠 Medium Risk of Lung Cancer")
        else:
            st.success("🟢 Low Risk of Lung Cancer")


if selection == "CNN Based Disease Prediction":

    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from keras.utils import image
    from tempfile import NamedTemporaryFile

    st.header("🧠 Lung Cancer Detection using CT Scan")

    @st.cache_resource
    def load_cnn_model():
        return load_model("models/keras_model.h5")

    cnn_model = load_cnn_model()

    uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        temp = NamedTemporaryFile(delete=False)
        temp.write(uploaded_file.getvalue())

        img = image.load_img(temp.name, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = cnn_model.predict(img_array)

        if prediction[0][0] >= 0.5:
            st.success("🟢 Normal Case")
        else:
            st.error("🔴 Lung Cancer Detected")
