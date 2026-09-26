@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"
set "HF_HOME=%~dp0..\models\huggingface-home"
set "HF_HUB_VERBOSITY=error"

if not exist ".venv\Scripts\python.exe" (
  echo Dependencies are missing. Starting setup.bat...
  call setup.bat
  if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" src\wra.py
if errorlevel 1 (
  echo.
  echo Processing did not complete. Review the message above.
  pause
  exit /b 1
)

echo.
echo Processing completed. Timeline and local relationship report are in the output directory.
pause
