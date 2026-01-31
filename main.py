# app.py
from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel

# 1. Define the Input Schema (Data Validation)
class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: int
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: int
    TAX: int
    PTRATIO: float
    B: float
    LSTAT: float

# 2. Initialize FastAPI
app = FastAPI(title="Real Estate Price Predictor API")

# 3. Load the Model Pipeline (Global)
# This includes the Imputer + Random Forest
model_path = "artifacts/house_price_pipeline.pkl"
pipeline = joblib.load(model_path)

@app.get("/")
def home():
    return {"message": "Real Estate Price Prediction API is Running!"}

@app.post("/predict")
def predict_price(features: HouseFeatures):
    input_df = pd.DataFrame([features.dict()])
    
    # Structural Engineering
    input_df['TAX_PER_RM'] = input_df['TAX'] / (input_df['RM'] + 1)
    input_df['LSTAT'] = np.log1p(input_df['LSTAT'])
    
    # Reorder columns to match exactly what was seen during fit
    # (Since QuantileTransformer inside the pipeline expects a specific order)
    column_order = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'TAX_PER_RM']
    input_df = input_df[column_order]

    log_prediction = pipeline.predict(input_df)
    return {"predicted_price_k": round(float(np.expm1(log_prediction)[0]), 2)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)