import streamlit as st
import joblib
import numpy as np
import pandas as pd
import requests

model = joblib.load('model_ranfor_oop.pkl')
encoder = joblib.load('encoders_oop.pkl')

def main():
    st.title('Obesity Prediction - UAS Model Deployment')

    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=1, max_value=100)
    height = st.number_input("Height (in meters)", min_value=1.0, max_value=2.5)
    weight = st.number_input("Weight (in kg)", min_value=20.0, max_value=300.0)
    family_history = st.selectbox("Family history with overweight", ["yes", "no"])
    favc = st.selectbox("Frequent consumption of high caloric food (FAVC)", ["yes", "no"])
    fcvc = st.slider("Frequency of consumption of vegetables (FCVC)", 1.0, 3.0, 2.0)
    ncp = st.slider("Number of main meals (NCP)", 1.0, 4.0, 3.0)
    caec = st.selectbox("Consumption of food between meals (CAEC)", ["Always", "Frequently", "Sometimes", "no"])
    smoke = st.selectbox("Do you smoke? (SMOKE)", ["yes", "no"])
    ch2o = st.slider("Water consumption (CH2O) in liters", 1.0, 3.0, 3.0)
    scc = st.selectbox("Do you monitor your calories (SCC)?", ["yes", "no"])
    faf = st.slider("Physical activity frequency (FAF)", 0.0, 3.0, 3.0)
    tue = st.slider("Time using technology devices (TUE)", 0.0, 2.0, 2.0)
    calc = st.selectbox("Do you drink alcohol? (CALC)", ["Always", "Frequently", "Sometimes", "no"])
    mtrans = st.selectbox("Transportation used (MTRANS)", ["Public_Transportation", "Walking", "Bike", "Motorbike", "Automobile"])

    data = {
        'Gender': gender,
        'Age': int(age),
        'Height': float(height),
        'Weight': float(weight),
        'family_history_with_overweight': family_history,
        'FAVC': favc,
        'FCVC': float(fcvc),
        'NCP': float(ncp),
        'CAEC': caec,
        'SMOKE': smoke,
        'CH2O': float(ch2o),
        'SCC': scc,
        'FAF': float(faf),
        'TUE': float(tue),
        'CALC': calc,
        'MTRANS': mtrans
    }

    if st.button('Predict'):
        features=data     
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')


def make_prediction(features):
    #response=requests.post("http://127.0.0.1:8000/predict", json=features)
    #prediction = response.json()["prediction"]
    prediction = model.predict(features)
    label = encoder.inverse_transform(prediction)
    return label[0]


if __name__ == '__main__':
    main()
