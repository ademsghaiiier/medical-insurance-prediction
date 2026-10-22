import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import os

st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="centered"
)

# Load model
model = tf.keras.models.load_model("model.keras")

# Load scaler only if exists
if not os.path.exists("scaler.pkl"):
    st.error(
        "scaler.pkl not found. Please run train_model.py first."
    )
    st.stop()

scaler = joblib.load("scaler.pkl")

st.title("🏥 Medical Insurance Cost Prediction Using Deep Learning")

st.write(
    "Predict medical insurance costs based on age, BMI, smoking status, region, and other customer information."
)

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

with col2:
    sex = st.selectbox(
        "Sex",
        ["Female", "Male"]
    )

bmi = st.slider(
    "BMI",
    10.0,
    60.0,
    25.0
)

children = st.slider(
    "Number of Children",
    0,
    10,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    [
        "northeast",
        "northwest",
        "southeast",
        "southwest"
    ]
)

st.markdown("---")

if st.button("Predict Insurance Cost"):

    sex_value = 0 if sex == "Female" else 1
    smoker_value = 1 if smoker == "Yes" else 0

    region_mapping = {
        "northeast": 0,
        "northwest": 1,
        "southeast": 2,
        "southwest": 3
    }

    region_value = region_mapping[region]

    data = np.array([[
        age,
        sex_value,
        bmi,
        children,
        smoker_value,
        region_value
    ]])

    data = scaler.transform(data)

    prediction = model.predict(
        data,
        verbose=0
    )

    st.success("Prediction Completed ✅")

    st.metric(
        "Estimated Insurance Cost",
        f"${prediction[0][0]:,.2f}"
    )

st.markdown("---")
st.caption("Developed by Adem Benalhadj Sghaier")
