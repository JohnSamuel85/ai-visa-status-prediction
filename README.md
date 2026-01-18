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
- `README.md` — Project documentation

---

## Tech Stack
- **Language:** Python
- **Data Handling:** Pandas, NumPy
- **EDA & Visualization:** Matplotlib, Seaborn
