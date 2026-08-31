@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe dashboard_controller.py
) else (
  python dashboard_controller.py
)
pause
