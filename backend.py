from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI()
model = joblib.load('model.pkl')
encoder = joblib.load('encoder.pkl')

class obese(BaseModel):
    Gender: str
    Age: int
    Height: float
    Weight: float
    family_history_with_overweight: str
    FAVC: str
    FCVC: float
    NCP: float
    CAEC: str
    SMOKE: str
    CH2O: float
    SCC: str
    FAF: float
    TUE: float
    CALC: str
    MTRANS: str

@app.get("/")
def read_root():
       return {"message": "Welcome to the ML Model API"}

@app.post('/predict')

def predict(obs: obese):
    data_in = obs.dict()
    features = pd.DataFrame([data_in])
    prediction = model.predict(features)
    label = encoder.inverse_transform([int(prediction[0])])[0]
    return {'prediction': label}