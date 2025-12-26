import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ----------------------------
# Load Dataset
# ----------------------------
# Make sure winequality-red.csv is in the same folder
data = pd.read_csv("winequality-red.csv")

X = data.drop("quality", axis=1)
y = data["quality"]

# ----------------------------
# Train Model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("🍷 Wine Quality Prediction App")
st.write("Enter the chemical properties of wine to predict its quality")

# Input fields
fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.number_input("Citric Acid", 0.0, 1.0, 0.0)
residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.076)
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", 0, 100, 11)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", 0, 300, 34)
density = st.number_input("Density", 0.9900, 1.0100, 0.9978)
pH = st.number_input("pH", 2.0, 4.5, 3.51)
sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.number_input("Alcohol", 0.0, 20.0, 9.4)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict Wine Quality"):
    sample_data = np.array([
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]).reshape(1, -1)

    prediction = model.predict(sample_data)

    st.success(f"🍷 Predicted Wine Quality: {prediction[0]}")

