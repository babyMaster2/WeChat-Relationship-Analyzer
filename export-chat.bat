@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Dependencies are missing. Starting setup.bat...
  call setup.bat
  if errorlevel 1 exit /b 1
)
".venv\Scripts\python.exe" src\export_wechat4.py
pause
