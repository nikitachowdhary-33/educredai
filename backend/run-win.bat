@echo off
REM Activate venv (assumes venv in repository root or adjust path)
if exist ".venv\Scripts\activate.bat" (
  call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
  call venv\Scripts\activate.bat
) else (
  echo "No virtual environment activation script found in .venv\\ or venv\\ — start one or run 'python -m venv venv' first"
)

pip install -r requirements.txt
python app.py
pause
