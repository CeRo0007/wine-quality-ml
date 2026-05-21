import streamlit as st
import numpy as np
import joblib

modelo = joblib.load("modelos/modelo_random.pkl")
scaler = joblib.load("modelos/scaler.pkl")

st.title("Predicción de Calidad del Vino")

st.write("Nombre: TU NOMBRE")
st.write("Código ISIL: TU CODIGO")

fixed_acidity = st.number_input("Fixed Acidity", 0.0, 20.0, 7.0)

volatile_acidity = st.number_input("Volatile Acidity", 0.0, 2.0, 0.5)

citric_acid = st.number_input("Citric Acid", 0.0, 2.0, 0.3)

residual_sugar = st.number_input("Residual Sugar", 0.0, 20.0, 2.0)

chlorides = st.number_input("Chlorides", 0.0, 1.0, 0.05)

free_sulfur_dioxide = st.number_input(
    "Free Sulfur Dioxide",
    0.0, 100.0, 15.0
)

total_sulfur_dioxide = st.number_input(
    "Total Sulfur Dioxide",
    0.0, 300.0, 50.0
)

density = st.number_input("Density", 0.9, 1.1, 0.99)

pH = st.number_input("pH", 0.0, 14.0, 3.2)

sulphates = st.number_input("Sulphates", 0.0, 2.0, 0.6)

alcohol = st.number_input("Alcohol", 0.0, 20.0, 10.0)

wine_type = st.selectbox(
    "Wine Type",
    ["Red", "White"]
)

type_white = 1 if wine_type == "White" else 0

if st.button("Predecir"):

    datos = np.array([[
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
        alcohol,
        type_white
    ]])

    datos = scaler.transform(datos)

    pred = modelo.predict(datos)

    if pred[0] == 1:
        st.success("Vino de Buena Calidad")
    else:
        st.error("Vino de Baja Calidad")
