"""
AI Visa Prediction Platform - ML Model Training Script
Generates synthetic dataset and trains Linear Regression + Random Forest models
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

np.random.seed(42)
N = 12000

# --- Country Risk Scores (higher = more risk of rejection) ---
NATIONALITIES = {
    'United States': 0.05, 'United Kingdom': 0.07, 'Canada': 0.06, 'Australia': 0.06,
    'Germany': 0.07, 'France': 0.08, 'Japan': 0.06, 'South Korea': 0.08,
    'India': 0.35, 'China': 0.30, 'Pakistan': 0.55, 'Bangladesh': 0.50,
    'Nigeria': 0.60, 'Ghana': 0.45, 'Ethiopia': 0.55, 'Kenya': 0.40,
    'Brazil': 0.25, 'Mexico': 0.28, 'Colombia': 0.35, 'Venezuela': 0.65,
    'Russia': 0.30, 'Ukraine': 0.25, 'Turkey': 0.28, 'Iran': 0.70,
    'Afghanistan': 0.80, 'Syria': 0.78, 'Iraq': 0.75, 'Yemen': 0.78,
    'Philippines': 0.30, 'Indonesia': 0.28, 'Vietnam': 0.30, 'Thailand': 0.20,
    'Saudi Arabia': 0.15, 'UAE': 0.12, 'Qatar': 0.10, 'Egypt': 0.40,
    'Morocco': 0.38, 'Algeria': 0.45, 'Tunisia': 0.38, 'Senegal': 0.42,
    'Argentina': 0.18, 'Chile': 0.15, 'Peru': 0.28, 'Ecuador': 0.33,
    'Poland': 0.10, 'Romania': 0.15, 'Bulgaria': 0.18, 'Hungary': 0.12,
    'Malaysia': 0.18, 'Sri Lanka': 0.35, 'Nepal': 0.40, 'Myanmar': 0.45,
}

DESTINATIONS = {
    'United States': {'base_days': 15, 'strictness': 0.4},
    'United Kingdom': {'base_days': 10, 'strictness': 0.35},
    'Canada': {'base_days': 12, 'strictness': 0.3},
    'Australia': {'base_days': 14, 'strictness': 0.3},
    'Schengen (EU)': {'base_days': 10, 'strictness': 0.25},
    'Germany': {'base_days': 8, 'strictness': 0.2},
    'France': {'base_days': 8, 'strictness': 0.22},
    'Japan': {'base_days': 5, 'strictness': 0.2},
    'UAE': {'base_days': 3, 'strictness': 0.15},
    'Singapore': {'base_days': 3, 'strictness': 0.12},
    'New Zealand': {'base_days': 10, 'strictness': 0.25},
    'Switzerland': {'base_days': 10, 'strictness': 0.22},
    'Netherlands': {'base_days': 8, 'strictness': 0.2},
    'Sweden': {'base_days': 8, 'strictness': 0.18},
    'Norway': {'base_days': 8, 'strictness': 0.2},
    'Denmark': {'base_days': 7, 'strictness': 0.18},
    'Austria': {'base_days': 8, 'strictness': 0.2},
    'Italy': {'base_days': 10, 'strictness': 0.25},
    'Spain': {'base_days': 9, 'strictness': 0.22},
    'Portugal': {'base_days': 8, 'strictness': 0.2},
    'Ireland': {'base_days': 8, 'strictness': 0.22},
    'Belgium': {'base_days': 7, 'strictness': 0.2},
    'South Korea': {'base_days': 5, 'strictness': 0.18},
    'Taiwan': {'base_days': 5, 'strictness': 0.15},
    'Thailand': {'base_days': 2, 'strictness': 0.1},
    'Malaysia': {'base_days': 1, 'strictness': 0.08},
    'Turkey': {'base_days': 1, 'strictness': 0.1},
    'China': {'base_days': 7, 'strictness': 0.3},
    'India': {'base_days': 5, 'strictness': 0.2},
    'South Africa': {'base_days': 7, 'strictness': 0.2},
}

VISA_TYPES = ['Tourist', 'Student', 'Work', 'Business', 'Transit', 'Family/Spouse', 'Immigrant/PR']
VISA_TYPE_DIFFICULTY = {'Tourist': 0.2, 'Transit': 0.1, 'Business': 0.25, 
                         'Family/Spouse': 0.3, 'Student': 0.35, 'Work': 0.4, 'Immigrant/PR': 0.55}
VISA_TYPE_DAYS_FACTOR  = {'Tourist': 1.0, 'Transit': 0.5, 'Business': 1.2, 
                           'Family/Spouse': 1.5, 'Student': 2.0, 'Work': 2.5, 'Immigrant/PR': 4.0}

GENDERS = ['Male', 'Female', 'Other']
EDUCATION_LEVELS = ['No Formal Education', 'High School', 'Associate Degree', 'Bachelor\'s Degree', 
                    'Master\'s Degree', 'PhD', 'Professional Degree']
EMPLOYMENT_STATUSES = ['Unemployed', 'Self-Employed', 'Private Sector', 'Government', 'Student', 'Retired']
TRAVEL_PURPOSES = ['Tourism', 'Business Meeting', 'Education', 'Family Visit', 'Medical', 
                   'Conference', 'Work Assignment', 'Permanent Residence']
LANGUAGES = ['None', 'Basic', 'Intermediate', 'Advanced', 'Native']

def generate_dataset(n=N):
    nat_list = list(NATIONALITIES.keys())
    dest_list = list(DESTINATIONS.keys())
    visa_types = VISA_TYPES
    
    data = []
    for _ in range(n):
        nationality = np.random.choice(nat_list)
        destination = np.random.choice(dest_list)
        visa_type = np.random.choice(visa_types)
        gender = np.random.choice(GENDERS, p=[0.49, 0.49, 0.02])
        age = int(np.random.beta(3, 2) * 65 + 18)
        age = min(max(age, 18), 80)
        
        education = np.random.choice(EDUCATION_LEVELS, p=[0.03, 0.20, 0.08, 0.35, 0.20, 0.08, 0.06])
        employment = np.random.choice(EMPLOYMENT_STATUSES, p=[0.10, 0.15, 0.35, 0.15, 0.15, 0.10])
        purpose = np.random.choice(TRAVEL_PURPOSES)
        language = np.random.choice(LANGUAGES, p=[0.05, 0.15, 0.30, 0.30, 0.20])
        
        annual_income = max(0, np.random.lognormal(10.2, 1.0))
        bank_balance = max(0, annual_income * np.random.uniform(0.1, 3.0))
        travel_history = int(np.random.exponential(3))
        previous_rejections = int(np.clip(np.random.exponential(0.3), 0, 5))
        accommodation_proof = np.random.choice([0, 1], p=[0.2, 0.8])
        return_ticket = np.random.choice([0, 1], p=[0.15, 0.85])
        invitation_letter = np.random.choice([0, 1], p=[0.4, 0.6])
        health_insurance = np.random.choice([0, 1], p=[0.25, 0.75])
        
        # --- Compute approval probability ---
        risk = NATIONALITIES[nationality]
        dest_info = DESTINATIONS[destination]
        visa_diff = VISA_TYPE_DIFFICULTY[visa_type]
        
        edu_score = EDUCATION_LEVELS.index(education) / (len(EDUCATION_LEVELS) - 1)
        lang_score = LANGUAGES.index(language) / (len(LANGUAGES) - 1)
        income_norm = min(annual_income / 100000, 1.0)
        balance_norm = min(bank_balance / 50000, 1.0)
        age_score = 1.0 - abs(age - 35) / 35
        
        p_approve = (
            (1 - risk) * 0.25 +
            (1 - dest_info['strictness']) * 0.15 +
            (1 - visa_diff) * 0.15 +
            edu_score * 0.10 +
            lang_score * 0.08 +
            income_norm * 0.10 +
            balance_norm * 0.07 +
            age_score * 0.03 +
            travel_history * 0.01 +
            accommodation_proof * 0.02 +
            return_ticket * 0.02 +
            health_insurance * 0.01 +
            invitation_letter * 0.01 -
            previous_rejections * 0.08
        )
        p_approve = np.clip(p_approve + np.random.normal(0, 0.07), 0.02, 0.98)
        approval_status = int(np.random.random() < p_approve)
        
        # --- Compute processing days ---
        base = dest_info['base_days']
        factor = VISA_TYPE_DAYS_FACTOR[visa_type]
        days = base * factor
        days += risk * 20
        days += dest_info['strictness'] * 15
        days += previous_rejections * 3
        days -= travel_history * 0.3
        days -= edu_score * 2
        days = max(1, days + np.random.normal(0, abs(days) * 0.2))
        processing_days = int(round(days))
        
        data.append({
            'nationality': nationality,
            'destination_country': destination,
            'visa_type': visa_type,
            'gender': gender,
            'age': age,
            'education_level': education,
            'employment_status': employment,
            'travel_purpose': purpose,
            'language_proficiency': language,
            'annual_income': round(annual_income, 2),
            'bank_balance': round(bank_balance, 2),
            'travel_history_count': travel_history,
            'previous_rejections': previous_rejections,
            'accommodation_proof': accommodation_proof,
            'return_ticket': return_ticket,
            'invitation_letter': invitation_letter,
            'health_insurance': health_insurance,
            'approval_status': approval_status,
            'processing_days': processing_days,
        })
    return pd.DataFrame(data)

def train_and_save():
    print("Generating dataset...")
    df = generate_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Approval rate: {df['approval_status'].mean():.2%}")
    print(f"Avg processing days: {df['processing_days'].mean():.1f}")
    
    # Save dataset
    df.to_csv('models/visa_dataset.csv', index=False)
    
    # Encode categorical features
    cat_cols = ['nationality', 'destination_country', 'visa_type', 'gender', 
                'education_level', 'employment_status', 'travel_purpose', 'language_proficiency']
    
    encoders = {}
    df_encoded = df.copy()
    for col in cat_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    feature_cols = ['nationality', 'destination_country', 'visa_type', 'gender', 'age',
                    'education_level', 'employment_status', 'travel_purpose', 'language_proficiency',
                    'annual_income', 'bank_balance', 'travel_history_count', 'previous_rejections',
                    'accommodation_proof', 'return_ticket', 'invitation_letter', 'health_insurance']
    
    X = df_encoded[feature_cols]
    y_class = df_encoded['approval_status']
    y_reg = df_encoded['processing_days']
    
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=42
    )
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_cls_train)
    rf_acc = rf.score(X_test, y_cls_test)
    print(f"Random Forest Accuracy: {rf_acc:.4f}")
    
    # Train Linear Regression for processing days
    print("Training Linear Regression for processing days...")
    scaler = StandardScaler()
    lr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LinearRegression())
    ])
    lr_pipeline.fit(X_train, y_reg_train)
    lr_r2 = lr_pipeline.score(X_test, y_reg_test)
    print(f"Linear Regression R²: {lr_r2:.4f}")
    
    # Save models and encoders
    print("\nSaving models...")
    joblib.dump(rf, 'models/rf_model.pkl')
    joblib.dump(lr_pipeline, 'models/lr_model.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(feature_cols, 'models/feature_cols.pkl')
    
    # Save metadata for API
    import json
    metadata = {
        'rf_accuracy': rf_acc,
        'lr_r2': lr_r2,
        'nationalities': list(NATIONALITIES.keys()),
        'destinations': list(DESTINATIONS.keys()),
        'visa_types': VISA_TYPES,
        'genders': GENDERS,
        'education_levels': EDUCATION_LEVELS,
        'employment_statuses': EMPLOYMENT_STATUSES,
        'travel_purposes': TRAVEL_PURPOSES,
        'language_proficiency': LANGUAGES,
    }
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ All models saved successfully!")
    print(f"   RF Accuracy: {rf_acc:.4f}")
    print(f"   LR R²: {lr_r2:.4f}")
    return metadata

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train_and_save()
