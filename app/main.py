from fastapi import FastAPI
from schemas.Pydantic import data as DataSchema
from data.data import load_data
import pickle
import numpy  as np
# import logging 
# logger=logging(bas)
app=FastAPI()
@app.get('/')
def Home():
    return {''' message''':''' Credit Card Fraud Detection API

## Problem Statement

Credit card fraud is a significant challenge in the financial industry, causing billions of dollars in losses every year. Detecting fraudulent transactions in real time is difficult because fraudulent activities represent only a tiny fraction of all transactions and often exhibit patterns that are hard to identify manually.

This project uses Machine Learning techniques to analyze transaction features and classify transactions as either:

* **0 : Legitimate Transaction**
* **1 : Fraudulent Transaction**

The model has been trained on an anonymized credit card transaction dataset containing PCA-transformed features (V1–V28), along with transaction time and amount information.

## How to Use This API

### 1. Health Check

Verify that the API is running:

`GET /health`

### 2. Single Prediction

Send a transaction record to:

`POST /predict`

The API returns:

* Predicted Class (Fraud or Legitimate)
* Fraud Probability Score
* Risk Level

### 3. Batch Prediction

Upload a CSV file containing transaction records to:

`POST /batch_predict`

The API will:

* Process all transactions
* Flag suspicious transactions
* Return prediction results for each record
* Allow downloading the processed output

### Sample Workflow

Upload Transaction Data
→ Preprocessing and Validation
→ Machine Learning Prediction
→ Fraud Probability Estimation
→ Risk Assessment and Results

## Note

The features V1–V28 are anonymized PCA-transformed variables provided by the original dataset and are generated automatically by banking systems. They are not intended for manual user entry. Therefore, this application supports API-based inference and CSV batch processing for practical usage scenarios.

---

**Built with FastAPI and Machine Learning for scalable and production-ready fraud detection.**'''}
@app.post('/prediction')
def input_d(row_number:int):
    df=load_data()
    with open('model/model.pkl','rb') as f:
        model=pickle.load(f)
    x=df.iloc[row_number]
    X=np.array(x).reshape(1,-1)
    prediction=model.predict(X)[0]
    prediction_probab=model.predict_proba(X)[0][1]
    return {"prediction":int(prediction),
            "probability":float(prediction_probab)}



    





# @app.post(data:transacation)
