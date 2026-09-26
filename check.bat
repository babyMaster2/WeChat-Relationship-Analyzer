@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Dependencies are missing. Run setup.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" src\wra.py --check
pause
