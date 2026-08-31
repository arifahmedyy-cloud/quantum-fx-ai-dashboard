@echo off
cd /d "%~dp0"
python -m pip install -r dashboard_api_requirements.txt
python dashboard_api.py
pause
