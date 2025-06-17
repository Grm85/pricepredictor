# pricepredictor

# 💻 Laptop Price Predictor

This project is a machine learning-powered web app that predicts the price of a laptop based on its specifications. Built using **Python**, **scikit-learn**, and **Streamlit**, this app allows users to input features like brand, RAM, storage, screen size, CPU, GPU, and operating system to get an estimated laptop price.

---

## 🚀 Features

- Trained using a **Random Forest Regressor** for accurate price predictions.
- Intuitive **Streamlit web interface** for easy interaction.
- Handles both **numerical** and **categorical** inputs through preprocessing pipelines.
- Supports input features like:
  - Company
  - Type (Ultrabook, Gaming, Notebook, etc.)
  - Screen Size
  - Screen Resolution
  - CPU
  - RAM
  - Memory
  - GPU
  - Operating System
  - Weight

---

## 🧠 Machine Learning Workflow

- Data Cleaning & Preprocessing (e.g., parsing GB/TB values)
- Feature Transformation using `ColumnTransformer`
- Model Training with `RandomForestRegressor`
- Pipeline saved using `joblib`
- Model served via a `streamlit` app

---



 Author
Garima Choudhary
Asp Data Scientist 
