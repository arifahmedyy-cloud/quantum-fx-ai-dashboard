@echo off
title Quantum FX AI - New Dashboard
cd /d "%~dp0"

rem Stop the old Streamlit window if it is still open from an older build.
taskkill /FI "WINDOWTITLE eq Gold Bot - Running*" /T /F >nul 2>&1

if not exist "venv\Scripts\python.exe" (
  echo Python environment missing. Run Setup.bat first.
  pause
  exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
  echo Node.js LTS is required. Run Setup.bat after installing Node.js.
  pause
  exit /b 1
)

if not exist "frontend\node_modules" (
  echo Installing React dependencies...
  cd frontend
  call npm install
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
  cd ..
)

start "Quantum FX API" cmd /k "cd /d ""%~dp0"" && venv\Scripts\python.exe -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8808"
timeout /t 2 /nobreak >nul
start "Quantum FX React" cmd /k "cd /d ""%~dp0frontend"" && npm run dev -- --host 127.0.0.1"
timeout /t 2 /nobreak >nul
start "" "http://127.0.0.1:5173"

echo.
echo New React dashboard started.
echo Close the API/React windows to stop the dashboard services.
pause
