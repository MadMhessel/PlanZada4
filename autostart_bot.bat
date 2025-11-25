@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem Navigate to script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

rem Choose available Python interpreter (prefers py launcher if present)
set "PY_EXE="
where py >nul 2>&1 && set "PY_EXE=py"
if "%PY_EXE%"=="" (
    where python >nul 2>&1 && set "PY_EXE=python"
)
if "%PY_EXE%"=="" (
    echo [ERROR] Python 3.11+ is required. Install it and rerun this script.
    exit /b 1
)

rem Create virtual environment if missing
set "VENV_DIR=%SCRIPT_DIR%\.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [INFO] Creating virtual environment...
    %PY_EXE% -m venv "%VENV_DIR%"
)

rem Activate environment and install dependencies
call "%VENV_DIR%\Scripts\activate.bat"
python -m pip install --upgrade pip >nul
python -m pip install -r requirements.txt

rem Prepare environment file on first launch
if not exist ".env" (
    echo [INFO] .env not found. Creating a template...
    (
        echo TELEGRAM_TOKEN=
        echo GOOGLE_SHEETS_ID=
        echo GOOGLE_CALENDAR_ID=
        echo GOOGLE_PROJECT_ID=
        echo GOOGLE_APPLICATION_CREDENTIALS=service_account.json
        echo DIALOG_LOG_PATH=dialog_log.jsonl
        echo GENAI_MODEL=gemini-1.5-pro-latest
    )>".env"
    echo Fill in required values inside .env and rerun the script.
    exit /b 0
)

set "PYTHONPATH=%SCRIPT_DIR%;%PYTHONPATH%"
echo [INFO] Starting Telegram AI Secretary bot...
python main.py
