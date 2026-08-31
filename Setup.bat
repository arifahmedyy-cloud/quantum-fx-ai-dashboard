@echo off
title Quantum FX AI - Setup
cd /d "%~dp0"
echo ============================================
echo   Quantum FX AI - One Time Setup
echo ============================================
echo.

python -m venv venv
if errorlevel 1 (
  echo ERROR: Python 3.10+ is required.
  pause
  exit /b 1
)

call venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto :fail
pip install -r requirements.txt
if errorlevel 1 goto :fail

where node >nul 2>nul
if errorlevel 1 (
  echo.
  echo ERROR: Node.js LTS is required for the new React dashboard.
  echo Install Node.js LTS, then run Setup.bat again.
  pause
  exit /b 1
)

cd frontend
call npm install
if errorlevel 1 goto :fail
cd ..

if not exist ".env" (
  copy .env.example .env >nul
)

echo.
echo ============================================
echo   Setup COMPLETE
echo   New React dashboard only.
echo   Yahoo Finance dependency removed.
echo ============================================
pause
exit /b 0

:fail
echo.
echo SETUP FAILED. Check the error above.
pause
exit /b 1
