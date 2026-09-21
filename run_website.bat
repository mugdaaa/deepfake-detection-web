@echo off
title VeritasAI Deepfake Detection Lab
echo ========================================================
echo   VeritasAI // Deepfake Detection & Explainability Lab
echo   FaceForensics++ AI Vision Studio
echo ========================================================
echo.
cd /d "%~dp0"

echo Starting FastAPI server at http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
"C:\Users\new\AppData\Local\Programs\Python\Python311\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
