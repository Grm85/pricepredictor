import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("laptop_price_model.pkl")

st.title("💻 Laptop Price Predictor")

# Input fields
company = st.selectbox("Company", ['Dell', 'HP', 'Apple', 'Asus', 'Lenovo', 'Acer', 'MSI', 'Toshiba'])
typename = st.selectbox("Type", ['Ultrabook', 'Notebook', 'Gaming', '2 in 1 Convertible', 'Workstation'])
inches = st.number_input("Screen Size (Inches)", min_value=10.0, max_value=20.0, value=15.6)
screen_resolution = st.selectbox("Screen Resolution", ['1920x1080', '1366x768', '1600x900', '3840x2160'])
cpu = st.selectbox("CPU", ['Intel Core i7', 'Intel Core i5', 'Intel Core i3', 'AMD Ryzen 5', 'AMD Ryzen 7'])
ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
memory = st.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048])
gpu = st.selectbox("GPU", ['Intel HD Graphics', 'Nvidia GTX 1050', 'Nvidia GTX 1650', 'AMD Radeon', 'Nvidia RTX 3060'])
opsys = st.selectbox("Operating System", ['Windows 10', 'macOS', 'Linux', 'No OS'])
weight = st.number_input("Weight (kg)", min_value=0.5, max_value=5.0, value=2.0)

# Predict
if st.button("Predict Price"):
    input_data = pd.DataFrame([[company, typename, inches, screen_resolution, cpu, ram, memory, gpu, opsys, weight]],
                              columns=['Company', 'TypeName', 'Inches', 'ScreenResolution', 'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight'])

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Laptop Price: ₹ {prediction:,.2f}")
