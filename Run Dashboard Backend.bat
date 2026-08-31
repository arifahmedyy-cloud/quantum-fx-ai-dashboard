@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8808
) else (
  python -m uvicorn dashboard_api:app --host 127.0.0.1 --port 8808
)
pause
