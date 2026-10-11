@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo ERROR: Virtual environment not found at .venv\Scripts\python.exe
  echo Run the setup instructions in MILESTONE6_SETUP.md first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 pause
