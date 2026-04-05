from  fastapi import FastAPI
import uvicorn
import joblib
import pandas as pd
from src.schema.schema import BankData

app = FastAPI()

model=joblib.load("models/best_model.pkl")

@app.get("/")
def home():
    return{"message":"fraud detection"}

@app.post("/predict")
def predict(data:BankData):
    input_data=pd.DataFrame([data.model_dump()])
    prediction=model.predict(input_data)

    if prediction[0]==0:
        result="It is not a Fraud"
    else:
        result="It is a Fraud"

    return {"prediction":result}