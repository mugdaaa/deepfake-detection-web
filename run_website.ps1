Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  VeritasAI // Deepfake Detection & Explainability Lab  " -ForegroundColor White
Write-Host "  FaceForensics++ AI Vision Studio                      " -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

$pythonExe = "C:\Users\new\AppData\Local\Programs\Python\Python311\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

Start-Process "http://127.0.0.1:8000"
& $pythonExe -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
