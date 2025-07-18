import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("pollution_level_prediction.pkl")

st.title("Forecasted AQI Category Prediction")

# Input CO_AQI value
co_aqi = st.number_input("Enter forecasted CO AQI", min_value=0.0)

if st.button("Predict Category"):
    prediction = model.predict([[co_aqi]])
    st.success(f"Predicted AQI Category: {prediction[0]}")
