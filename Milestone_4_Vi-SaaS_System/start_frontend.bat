@echo off
title Vi-SaaS - Frontend Server
color 0B
echo.
echo  ============================================
echo    Vi-SaaS - AI Visa Prediction Platform
echo    Frontend Dev Server
echo  ============================================
echo.
echo  [OK] Starting frontend on http://localhost:3000
echo.
echo  Open your browser at: http://localhost:3000
echo  Press Ctrl+C to stop the server.
echo.
cd /d "%~dp0frontend"
python -m http.server 3000
pause
