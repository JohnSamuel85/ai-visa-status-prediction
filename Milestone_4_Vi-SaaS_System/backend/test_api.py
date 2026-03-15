import requests
import json

payload = {
    "visa_type": "Tourist",
    "destination_country": "United States",
    "nationality": "India",
    "age": 30,
    "gender": "Male",
    "education_level": "Bachelor's Degree",
    "employment_status": "Private Sector",
    "travel_purpose": "Tourism",
    "language_proficiency": "Advanced",
    "annual_income": 80000,
    "bank_balance": 15000,
    "travel_history_count": 3,
    "previous_rejections": 0,
    "accommodation_proof": 1,
    "return_ticket": 1,
    "invitation_letter": 0,
    "health_insurance": 1,
    "approval_probability": 85.5,
    "predicted_days": 12,
    "status": "APPROVED",
    "risk_factors": [{"factor": "Slight risk", "impact": "low"}],
    "positive_factors": ["Good income", "No rejections"]
}

try:
    response = requests.post("http://localhost:3000/api/gemini-analysis", json=payload)
    data = response.json()
    if data.get("powered_by") == "gemini-2.5-flash":
        print("SUCCESS_GEMINI_IS_WORKING")
    else:
        print("FALLBACK_OR_ERROR:", data.get("error_note", data.get("powered_by")))
except Exception as e:
    print("REQUEST_FAILED:", e)
