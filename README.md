# AI Enabled Visa Status Prediction and Processing Time Estimator

## Project Overview
This project focuses on building an AI-based system to:
- Predict **Visa Status** (Approved / Rejected)
- Estimate **Visa Processing Time** (in days)

The project is being developed milestone-by-milestone. This repository currently includes work completed for **Milestone 1** and **Milestone 2**.

---

## Milestones Completed

### ✅ Milestone 1: Data Collection & Preprocessing
**Objective:** Build a clean, structured dataset ready for modeling.

**Work Done:**
- Loaded the visa dataset and validated its structure
- Checked and confirmed **no missing values**
- Identified categorical and numerical features
- Encoded categorical variables into numerical form
- Separated features and target variables:
  - `visa_status` (classification target)
  - `processing_days` (regression target)
- Prepared the dataset in a model-ready format

---

### ✅ Milestone 2: Exploratory Data Analysis (EDA) & Feature Engineering
**Objective:** Derive insights and engineer meaningful features for modeling.

**Work Done:**
- Performed EDA using visualizations (Matplotlib/Seaborn)
- Analyzed how **processing time** varies by:
  - Visa type
  - Destination country
  - Application month (seasonality)
- Generated correlation analysis for numerical features
- Engineered new features:
  - `seasonal_index` (peak vs non-peak months)
  - `country_avg_processing_days` (country-wise avg processing time)
  - `visa_type_avg_processing_days` (visa-type-wise avg processing time)

**Outcome:** Dataset is now enhanced with engineered features and ready for modeling.

---

### ✅ Milestone 3: Predictive Modeling (Regression)
**Objective:** Develop and test regression models to predict visa processing time.

**Work Done:**
- Performed train-test split to evaluate models on unseen data
- Applied **leakage-safe feature engineering** using training-set reference only
- Encoded categorical variables for model compatibility
- Trained baseline regression models:
  - **Linear Regression**
  - **Random Forest Regressor**
- Evaluated models using:
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
  - **R² Score**
- Fine-tuned Random Forest using **RandomizedSearchCV** to improve performance
- Selected the best-performing model based on evaluation metrics
- Performed **segment-wise evaluation by visa type** to verify consistent performance across categories

**Outcome:** A best-performing regression model is selected and ready for further integration and deployment in upcoming milestones.

## Dataset
- File: `ai_visa_prediction_synthetic_dataset.csv`
- Records: 3000
- Columns: 12 (before feature engineering)
- Targets:
  - `visa_status`
  - `processing_days`

> Note: A synthetic dataset is used due to limited availability of public application-level visa datasets with processing time details.

---

## Repository Contents
- `ai_visa_prediction_synthetic_dataset.csv` — Dataset
- `ai_visa_prediction_1.py` (or preprocessing script) — Milestone 1 preprocessing code
- `milestone_2_eda_feature_engineering.py` — Milestone 2 EDA + feature engineering code
- `milestone_3_regression_modeling.py` — Milestone 3 regression modeling + evaluation + tuning code
- `README.md` — Project documentation

---

## Tech Stack
- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **EDA & Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
