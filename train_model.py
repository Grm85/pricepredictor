import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Load dataset
df = pd.read_csv("laptops.csv")

# Clean and transform data
df['Ram'] = df['Ram'].str.replace('GB', '').astype(int)
df['Weight'] = df['Weight'].str.replace('kg', '').astype(float)
df['Price'] = df['Price'].astype(float)

# Combine Memory columns into single feature
df['Memory'] = df['Memory'].str.replace('GB', '').str.replace('TB', '000')
df['Memory'] = df['Memory'].str.extract('(\d+)', expand=False).astype(float)

# Target and features
X = df.drop("Price", axis=1)
y = df["Price"]

# Identify column types
categorical = ['Company', 'TypeName', 'ScreenResolution', 'Cpu', 'Gpu', 'OpSys']
numerical = ['Inches', 'Ram', 'Memory', 'Weight']

# Create pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical),
    ('num', StandardScaler(), numerical)
])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "laptop_price_model.pkl")

print("Model trained and saved successfully.")
