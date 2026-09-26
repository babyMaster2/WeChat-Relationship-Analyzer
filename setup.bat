@echo off
setlocal
cd /d "%~dp0"
set "HF_HOME=%~dp0..\models\huggingface-home"
set "HF_HUB_VERBOSITY=error"
set "PIP_CACHE_DIR=%~dp0..\caches\pip"

echo [1/5] Detecting Python 3.9+...
set "PYTHON_EXE="
where python >nul 2>nul && python --version >nul 2>nul && set "PYTHON_EXE=python"
if not defined PYTHON_EXE if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" set "PYTHON_EXE=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
if not defined PYTHON_EXE py -3 --version >nul 2>nul && set "PYTHON_EXE=py -3"
if not defined PYTHON_EXE (
  echo Python 3.9+ was not found. Install Python 3.12 x64, then run setup.bat again.
  pause
  exit /b 1
)

echo [2/5] Creating the private project environment...
if not exist ".venv\Scripts\python.exe" %PYTHON_EXE% -m venv .venv
if errorlevel 1 goto :failed

echo [3/5] Installing local transcription dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :failed

echo [4/5] Installing the fully local relationship-analysis runtime and model...
".venv\Scripts\python.exe" src\setup_local_analysis.py
if errorlevel 1 goto :failed

echo [5/5] Running dependency and pipeline checks...
".venv\Scripts\python.exe" src\wra.py --self-test
if errorlevel 1 goto :failed

echo.
echo Setup completed. Run export-chat.bat, then start.bat.
echo Whisper transcription and Qwen relationship analysis are fully local. Chats and audio are never uploaded.
pause
exit /b 0

:failed
echo.
echo Setup failed. Review the error above, then run setup.bat again.
pause
exit /b 1
