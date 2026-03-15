# ================================
# Milestone 1: Data Preprocessing
# AI Enabled Visa Status Prediction (Vi-SaaS)
# ================================

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Dataset
# ================================

# Update path to point to the common dataset location
dataset_path = os.path.join("..", "Milestone_1_Dataset", "visa_dataset.csv")
if not os.path.exists(dataset_path):
    print(f"Warning: Dataset not found at {dataset_path}. Using fallback 'visa_dataset.csv'.")
    dataset_path = "visa_dataset.csv"

df = pd.read_csv(dataset_path)

print(f"Dataset loaded successfully from: {dataset_path}")
print("Shape:", df.shape)

# 2. Initial Data Inspection
# ================================

print("\nDataset Info:")
df.info()

print("\nFirst 5 rows:")
print(df.head())

# 3. Missing Value Check
# ================================

missing_values = df.isnull().sum()

print("\nMissing Values per Column:")
print(missing_values)

if missing_values.sum() > 0:
    print("Warning: Dataset contains missing values. Handling them...")
    df = df.dropna()
else:
    print("No missing values found.")

# 4. Identify Feature Types (Relevant to Vi-SaaS)
# ================================

# Features used in the system's training pipeline
feature_cols = [
    'nationality', 'destination_country', 'visa_type', 'gender', 'age',
    'education_level', 'employment_status', 'travel_purpose', 'language_proficiency',
    'annual_income', 'bank_balance', 'travel_history_count', 'previous_rejections',
    'accommodation_proof', 'return_ticket', 'invitation_letter', 'health_insurance'
]

categorical_columns = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
numerical_columns = df[feature_cols].select_dtypes(include=["int64", "float64"]).columns.tolist()

print("\nSystem-Relevant Categorical Columns:")
print(categorical_columns)

print("\nSystem-Relevant Numerical Columns:")
print(numerical_columns)

# 5. Categorical Encoding
# ================================

df_encoded = df.copy()
label_encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    label_encoders[col] = le

print("\nCategorical encoding completed for system features")

# 6. Feature and Target Separation
# ================================

# System predicts both approval_status (Classification) and processing_days (Regression)
X = df_encoded[feature_cols]
y_classification = df_encoded["approval_status"]
y_regression = df_encoded["processing_days"]

print("\nFeature matrix shape:", X.shape)
print("Classification target (approval_status) shape:", y_classification.shape)
print("Regression target (processing_days) shape:", y_regression.shape)

# 7. Feature Scaling
# ================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nFeature scaling (StandardScaler) completed")
print("Scaled feature shape:", X_scaled.shape)

# 8. Final Output Confirmation
# ================================

print("\nMilestone 1 Completed Successfully")
print("Dataset is clean, structured, encoded, and relevant to the Vi-SaaS system.")
