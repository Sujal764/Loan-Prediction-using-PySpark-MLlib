import logging
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# -----------------------
# Logging Configuration
# -----------------------
logging.basicConfig(
    filename="app.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

logger = logging.getLogger(__name__)

# -----------------------
# Load trained model
# -----------------------
pipeline = joblib.load("model_loan_defaulter.pkl")
threshold = 0.39

# -----------------------
# Request Data Model
# -----------------------
class ClientData(BaseModel):
    NAME_CONTRACT_TYPE: str = Field(..., example="Cash loans")
    CODE_GENDER: str = Field(..., example="M")
    AMT_INCOME_TOTAL: float = Field(..., example=202500.0)
    AMT_CREDIT: float = Field(..., example=406597.5)
    AMT_GOODS_PRICE: float = Field(..., example=351000.0)
    NAME_TYPE_SUITE: str = Field(..., example="Unaccompanied")
    NAME_INCOME_TYPE: str = Field(..., example="Working")
    NAME_EDUCATION_TYPE: str = Field(..., example="Higher education")
    NAME_FAMILY_STATUS: str = Field(..., example="Single / not married")
    NAME_HOUSING_TYPE: str = Field(..., example="House / apartment")
    REGION_POPULATION_RELATIVE: float = Field(..., example=0.028663)
    DAYS_BIRTH: int = Field(..., example=-16765)
    DAYS_EMPLOYED: int = Field(..., example=-3650)
    DAYS_REGISTRATION: int = Field(..., example=-4500)
    OCCUPATION_TYPE: str = Field(..., example="Laborers")
    CNT_FAM_MEMBERS: int = Field(..., example=2)
    REGION_RATING_CLIENT_W_CITY: int = Field(..., example=2)
    OBS_30_CNT_SOCIAL_CIRCLE: float = Field(..., ge=0, le=30, example=1.0)
    DEF_30_CNT_SOCIAL_CIRCLE: float = Field(..., ge=0, le=30, example=0.0)

    @field_validator("AMT_INCOME_TOTAL")
    def income_check(cls, v):
        if v <= 0:
            raise ValueError("Income must be a positive number")
        return v

    @field_validator("CNT_FAM_MEMBERS")
    def family_members_check(cls, v):
        if v <= 0:
            raise ValueError("Family members must be at least 1")
        return v


# -----------------------
# FastAPI App
# -----------------------
app = FastAPI(title="🏦 Home Loan Defaulter Prediction API with Logging")

@app.get("/")
def home():
    logger.info("Health Check Triggered")
    return {"message": "🏦 Home Loan Defaulter Prediction API is running 🚀"}


@app.post("/predict")
async def predict(data: ClientData, request: Request):

    try:
        input_df = pd.DataFrame([data.model_dump()])

        proba = pipeline.predict_proba(input_df)[:, 1]
        pred = (proba >= threshold).astype(int)[0]

        response = {
            "Predicted Probability": round(float(proba[0]), 4),
            "Prediction Class": int(pred),
            "Decision": "High Risk" if pred == 1 else "Low Risk"
        }

        logger.info(f"Prediction request from {request.client.host}: "
                    f"Input={data.model_dump()} | Output={response}")

        return response

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {"error": "Prediction failed. Check logs for details."}
