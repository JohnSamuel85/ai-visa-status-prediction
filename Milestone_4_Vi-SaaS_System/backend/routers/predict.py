"""Prediction router - ML-based visa prediction"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List
from .auth import verify_token
import joblib
import numpy as np
import os
import json

router = APIRouter(prefix="/api", tags=["prediction"])

# Load models at startup
_models = {}

def load_models():
    global _models
    base = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    try:
        _models['rf'] = joblib.load(os.path.join(base, 'rf_model.pkl'))
        _models['lr'] = joblib.load(os.path.join(base, 'lr_model.pkl'))
        _models['encoders'] = joblib.load(os.path.join(base, 'encoders.pkl'))
        _models['feature_cols'] = joblib.load(os.path.join(base, 'feature_cols.pkl'))
        with open(os.path.join(base, 'metadata.json')) as f:
            _models['metadata'] = json.load(f)
        print("[OK] Models loaded successfully")
    except Exception as e:
        print(f"[WARN] Models not loaded: {e}. Run train_models.py first.")

load_models()

class PredictionInput(BaseModel):
    nationality: str
    destination_country: str
    visa_type: str
    gender: str
    age: int
    education_level: str
    employment_status: str
    travel_purpose: str
    language_proficiency: str
    annual_income: float
    bank_balance: float
    travel_history_count: int
    previous_rejections: int
    accommodation_proof: int  # 0 or 1
    return_ticket: int        # 0 or 1
    invitation_letter: int    # 0 or 1
    health_insurance: int     # 0 or 1

def safe_encode(encoder, value, col):
    """Encode a value, using 0 for unknown categories"""
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Unknown category - find closest
        classes = list(encoder.classes_)
        return 0

@router.post("/predict")
async def predict_visa(data: PredictionInput):
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded. Run train_models.py first.")
    
    encoders = _models['encoders']
    feature_cols = _models['feature_cols']
    
    cat_map = {
        'nationality': data.nationality,
        'destination_country': data.destination_country,
        'visa_type': data.visa_type,
        'gender': data.gender,
        'education_level': data.education_level,
        'employment_status': data.employment_status,
        'travel_purpose': data.travel_purpose,
        'language_proficiency': data.language_proficiency,
    }
    
    num_map = {
        'age': data.age,
        'annual_income': data.annual_income,
        'bank_balance': data.bank_balance,
        'travel_history_count': data.travel_history_count,
        'previous_rejections': data.previous_rejections,
        'accommodation_proof': data.accommodation_proof,
        'return_ticket': data.return_ticket,
        'invitation_letter': data.invitation_letter,
        'health_insurance': data.health_insurance,
    }
    
    row = []
    for col in feature_cols:
        if col in cat_map:
            row.append(safe_encode(encoders[col], cat_map[col], col))
        else:
            row.append(num_map.get(col, 0))
    
    X = np.array([row])
    
    # Random Forest prediction
    rf_proba = _models['rf'].predict_proba(X)[0]
    approval_probability = float(rf_proba[1])
    approval_class = int(_models['rf'].predict(X)[0])
    
    # Linear Regression for processing days
    predicted_days = float(_models['lr'].predict(X)[0])
    predicted_days = max(1, predicted_days)
    
    # Compute confidence interval (±20%)
    days_low = int(max(1, predicted_days * 0.8))
    days_high = int(predicted_days * 1.3)
    
    # Risk factor analysis
    risk_factors = []
    if data.previous_rejections > 0:
        risk_factors.append({"factor": "Previous visa rejections", "impact": "High", "severity": "danger"})
    if data.annual_income < 15000:
        risk_factors.append({"factor": "Low annual income", "impact": "Medium", "severity": "warning"})
    if data.bank_balance < 5000:
        risk_factors.append({"factor": "Insufficient bank balance", "impact": "Medium", "severity": "warning"})
    if data.travel_history_count == 0:
        risk_factors.append({"factor": "No prior travel history", "impact": "Low", "severity": "info"})
    if not data.health_insurance:
        risk_factors.append({"factor": "No health insurance", "impact": "Medium", "severity": "warning"})
    if not data.accommodation_proof:
        risk_factors.append({"factor": "No accommodation proof", "impact": "High", "severity": "danger"})
    if not data.return_ticket:
        risk_factors.append({"factor": "No return ticket", "impact": "High", "severity": "danger"})
    if not data.invitation_letter and data.visa_type in ['Business', 'Work', 'Family/Spouse']:
        risk_factors.append({"factor": "Missing invitation letter for visa type", "impact": "High", "severity": "danger"})
    
    positive_factors = []
    if data.travel_history_count > 5:
        positive_factors.append("Strong travel history")
    if data.annual_income > 50000:
        positive_factors.append("Strong financial profile")
    if data.education_level in ["Master's Degree", "PhD", "Professional Degree"]:
        positive_factors.append("Strong educational background")
    if data.language_proficiency in ["Advanced", "Native"]:
        positive_factors.append("Strong language proficiency")
    if data.previous_rejections == 0:
        positive_factors.append("Clean visa history")
    
    # Determine status label
    if approval_probability >= 0.75:
        status = "High Approval Chance"
        status_color = "green"
    elif approval_probability >= 0.50:
        status = "Moderate Approval Chance"
        status_color = "yellow"
    elif approval_probability >= 0.30:
        status = "Low Approval Chance"
        status_color = "orange"
    else:
        status = "Very Low Approval Chance"
        status_color = "red"
    
    return {
        "approval_probability": round(approval_probability * 100, 1),
        "approval_class": approval_class,
        "status": status,
        "status_color": status_color,
        "predicted_days": int(round(predicted_days)),
        "processing_range": {"min": days_low, "max": days_high},
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "model_confidence": round(max(rf_proba) * 100, 1),
        "input_summary": {
            "visa_type": data.visa_type,
            "destination": data.destination_country,
            "nationality": data.nationality,
        }
    }

@router.get("/history")
async def get_history(current_user: dict = Depends(verify_token)):
    PREDICTIONS_FILE = "data/predictions.json"
    if not os.path.exists(PREDICTIONS_FILE):
        return []
        
    try:
        with open(PREDICTIONS_FILE, "r") as f:
            all_predictions = json.load(f)
            return all_predictions.get(current_user["email"], [])
    except Exception as e:
        print(f"Error loading history: {e}")
        return []

@router.get("/metadata")
async def get_metadata():
    if 'metadata' not in _models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return _models['metadata']
