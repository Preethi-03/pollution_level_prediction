import streamlit as st
import pandas as pd
import joblib

# Load model and data
svm_model = joblib.load('pollution_model.pkl')
df = pd.read_csv('output.csv')   # Make sure it has City, Hour, Temp columns

st.title("Air Quality Index Prediction App")

# City selection
cities = df['City'].unique()
selected_city = st.selectbox("Select City", cities)

# Hour selection
hours = sorted(df['Hour'].unique())
selected_hour = st.selectbox("Select Hour", hours)

# Temperature input (from dataset or user)
temp_choice = st.radio("Do you want to enter temperature manually?", ["Yes", "Use Dataset Value"])

if temp_choice == "Yes":
    temperature = st.number_input("Enter Temperature (°C)", min_value=-10.0, max_value=50.0, step=0.1)
else:
    # Get default temperature from dataset for selected city and hour
    temperature = df[(df['City'] == selected_city) & (df['Hour'] == selected_hour)]['Temperature'].mean()
    st.write(f"Using average dataset temperature: **{temperature:.1f}°C**")

# (Optional) Show city info
st.write(f"Selected City: **{selected_city}**, Hour: **{selected_hour}:00**, Temperature: **{temperature}°C**")

# AQI Inputs
CO_AQI = st.number_input("Enter CO AQI", min_value=0.0)
NO2_AQI = st.number_input("Enter NO2 AQI", min_value=0.0)
NOx_AQI = st.number_input("Enter NOx AQI", min_value=0.0)

# Predict button
if st.button("Predict AQI Category"):
    features = [[CO_AQI, NO2_AQI, NOx_AQI]]  # Add hour and temperature later if included in training
    prediction = svm_model.predict(features)
    st.success(f"Predicted AQI Category: {prediction[0]}")
