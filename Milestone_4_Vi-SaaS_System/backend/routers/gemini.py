"""Gemini AI router - Advanced visa analysis using Gemini 2.5 Flash"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, List, Dict
from .auth import verify_token
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(prefix="/api", tags=["gemini"])

PREDICTIONS_FILE = "data/predictions.json"
os.makedirs("data", exist_ok=True)

def save_prediction(user_email: str, prediction_data: dict):
    all_predictions = {}
    if os.path.exists(PREDICTIONS_FILE):
        try:
            with open(PREDICTIONS_FILE, "r") as f:
                all_predictions = json.load(f)
        except: pass
    
    if user_email not in all_predictions:
        all_predictions[user_email] = []
    
    all_predictions[user_email].append(prediction_data)
    
    with open(PREDICTIONS_FILE, "w") as f:
        json.dump(all_predictions, f, indent=2)

try:
    import google.generativeai as genai
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY and GEMINI_API_KEY != "your_gemini_api_key_here":
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-2.5-flash")
        GEMINI_AVAILABLE = True
        print("[OK] Gemini 2.5 Flash configured")
    else:
        GEMINI_AVAILABLE = False
        print("[WARN] Gemini API key not configured - using fallback responses")
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARN] google-generativeai not installed")

class GeminiAnalysisRequest(BaseModel):
    visa_type: str
    destination_country: str
    nationality: str
    age: int
    gender: str
    education_level: str
    employment_status: str
    travel_purpose: str
    language_proficiency: str
    annual_income: float
    bank_balance: float
    travel_history_count: int
    previous_rejections: int
    accommodation_proof: int
    return_ticket: int
    invitation_letter: int
    health_insurance: int
    # ML Results
    approval_probability: float
    predicted_days: int
    status: str
    risk_factors: List[Dict]
    positive_factors: List[str]

def get_fallback_analysis(req: GeminiAnalysisRequest) -> dict:
    """Fallback when Gemini is not configured"""
    docs = []
    if req.visa_type == "Tourist":
        docs = ["Valid passport (6+ months validity)", "Proof of accommodation", "Return flight ticket", 
                "Bank statements (3 months)", "Travel insurance", "Itinerary"]
    elif req.visa_type == "Student":
        docs = ["Acceptance letter from institution", "Proof of funds", "Academic transcripts",
                "Language proficiency certificate", "Health insurance", "Accommodation letter"]
    elif req.visa_type == "Work":
        docs = ["Job offer letter", "Work permit", "Educational certificates", "Professional references",
                "Tax documents", "Proof of accommodation"]
    else:
        docs = ["Valid passport", "Completed application form", "Proof of funds", 
                "Supporting documents for visa type", "Passport photos", "Travel insurance"]
    
    tips = []
    if req.previous_rejections > 0:
        tips.append("Address previous rejection reasons explicitly in your new application.")
    if req.bank_balance < 5000:
        tips.append("Consider increasing your bank balance before applying — aim for at least $5,000.")
    if req.travel_history_count == 0:
        tips.append("Building travel history through neighboring countries can strengthen your application.")
    if not req.health_insurance:
        tips.append("Purchase comprehensive travel/health insurance — it significantly improves approval chances.")
    tips.append(f"Apply at least {req.predicted_days + 14} days before your travel date for buffer.")
    tips.append("Ensure all documents are translated to the destination country's official language if needed.")
    
    return {
        "summary": f"Based on your profile, your {req.visa_type} visa application to {req.destination_country} has a {req.approval_probability:.0f}% approval probability. {req.status}.",
        "document_checklist": docs,
        "ai_tips": tips,
        "timeline_advice": f"Expected processing: {req.predicted_days} days. We recommend submitting your application at least {req.predicted_days + 14} days before travel.",
        "risk_assessment": "High risk" if req.approval_probability < 40 else ("Moderate risk" if req.approval_probability < 65 else "Low risk"),
        "powered_by": "fallback"
    }

@router.post("/gemini-analysis")
async def gemini_analysis(req: GeminiAnalysisRequest, current_user: dict = Depends(verify_token)):
    result = None
    if not GEMINI_AVAILABLE:
        result = get_fallback_analysis(req)
    else:
        try:
            risk_list = "\n".join([f"- {r['factor']} ({r['impact']} impact)" for r in req.risk_factors]) or "None identified"
            positive_list = "\n".join([f"- {p}" for p in req.positive_factors]) or "None identified"
            
            prompt = f"""You are an expert visa consultant AI. Analyze the following visa application profile and provide detailed, actionable insights.

## Applicant Profile
- **Visa Type**: {req.visa_type}
- **Destination**: {req.destination_country}
- **Nationality**: {req.nationality}
- **Age**: {req.age} | **Gender**: {req.gender}
- **Education**: {req.education_level}
- **Employment**: {req.employment_status}
- **Travel Purpose**: {req.travel_purpose}
- **Language Proficiency**: {req.language_proficiency}
- **Annual Income**: ${req.annual_income:,.0f}
- **Bank Balance**: ${req.bank_balance:,.0f}
- **Travel History**: {req.travel_history_count} trips
- **Previous Rejections**: {req.previous_rejections}
- **Has Accommodation Proof**: {"Yes" if req.accommodation_proof else "No"}
- **Has Return Ticket**: {"Yes" if req.return_ticket else "No"}
- **Has Invitation Letter**: {"Yes" if req.invitation_letter else "No"}
- **Has Health Insurance**: {"Yes" if req.health_insurance else "No"}

## ML Model Results
- **Approval Probability**: {req.approval_probability:.1f}%
- **Status**: {req.status}
- **Estimated Processing Days**: {req.predicted_days} days

## Risk Factors Identified
{risk_list}

## Positive Factors
{positive_list}

Please provide a JSON response with EXACTLY this structure:
{{
  "summary": "2-3 sentence executive summary of the application",
  "document_checklist": ["doc1", "doc2", "doc3", "doc4", "doc5", "doc6"],
  "ai_tips": ["tip1", "tip2", "tip3", "tip4"],
  "timeline_advice": "Specific advice about timing the application",
  "risk_assessment": "Overall risk level: Low/Moderate/High/Critical",
  "specific_advice": "2-3 sentences of specific advice based on this exact profile"
}}

Be specific, actionable and tailored to this exact applicant profile. Return ONLY valid JSON."""

            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.4,
                    max_output_tokens=1500,
                )
            )
            
            import re
            text = response.text.strip()
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['powered_by'] = 'gemini-2.5-flash'
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"Gemini error: {e}")
            result = get_fallback_analysis(req)
            result['error_note'] = str(e)

    # Persistence: Save the complete prediction result
    if result:
        full_record = {
            "id": str(uuid.uuid4()),
            "date": datetime.utcnow().isoformat(),
            "visaType": req.visa_type,
            "destination": req.destination_country,
            "nationality": req.nationality,
            "gender": req.gender,
            "age": req.age,
            "approval_probability": req.approval_probability,
            "predicted_days": req.predicted_days,
            "status": req.status,
            "status_color": "green" if req.approval_probability >= 75 else ("yellow" if req.approval_probability >= 50 else ("orange" if req.approval_probability >= 30 else "red")),
            "risk_factors": req.risk_factors,
            "positive_factors": req.positive_factors,
            "gemini": result
        }
        save_prediction(current_user["email"], full_record)
        return result
    
    raise HTTPException(status_code=500, detail="Failed to generate or save analysis")
