@echo off
title Vi-SaaS - Backend Setup & Launch
color 0A
echo.
echo  ============================================
echo    Vi-SaaS - AI Visa Prediction Platform
echo    Backend Setup ^& Launch Script
echo  ============================================
echo.

REM --- Find Python ---
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo  [ERROR] Python not found. Please install Python 3.9+ from:
    echo           https://www.python.org/downloads/
    echo           Make sure to check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)

echo  [OK] Python found:
python --version
echo.

REM --- Install dependencies ---
echo  [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo  [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)
echo.

REM --- Train models ---
echo  [2/3] Training ML models (this takes ~30 seconds)...
if not exist "models\rf_model.pkl" (
    python train_models.py
    if %errorlevel% neq 0 (
        echo  [ERROR] Model training failed.
        pause
        exit /b 1
    )
) else (
    echo  [OK] Models already trained, skipping...
)
echo.

REM --- Start server ---
echo  [3/3] Starting FastAPI server on http://localhost:5000
echo.
echo  ============================================
echo    API Docs: http://localhost:5000/docs
echo    Health:   http://localhost:5000/health
echo  ============================================
echo.
python -m uvicorn main:app --host 0.0.0.0 --port 5000 --reload
pause
