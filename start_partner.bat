@echo off
setlocal
cd /d "%~dp0"
title SignalGuard - Partner Setup

set "PYTHONUTF8=1"
set "SIGNALGUARD_LIGHT_MODE=1"
set "VENV=%~dp0.venv"
set "PYTHON=%VENV%\Scripts\python.exe"
set "READY=%VENV%\.signalguard_partner_ready"

echo.
echo ==========================================
echo          SignalGuard Partner Setup
echo ==========================================
echo.

if not exist "%PYTHON%" (
    echo [1/3] Creating local Python environment...
    where python >nul 2>&1
    if errorlevel 1 (
        echo.
        echo Python was not found.
        echo Install Python 3.12+ from https://www.python.org/
        echo Then run this file again.
        pause
        exit /b 1
    )
    python -m venv "%VENV%"
    if errorlevel 1 (
        echo.
        echo Could not create the Python environment.
        pause
        exit /b 1
    )
)

if not exist "%READY%" (
    echo [2/3] Installing lightweight SignalGuard dependencies...
    echo This happens only on the first run.
    echo.

    "%PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo.
        echo Failed while updating pip.
        pause
        exit /b 1
    )

    "%PYTHON%" -m pip install -r "%~dp0requirements.partner.txt"
    if errorlevel 1 (
        echo.
        echo Failed while installing dependencies.
        pause
        exit /b 1
    )

    type nul > "%READY%"
)

echo [3/3] Starting SignalGuard...
echo.
echo Partner mode does NOT require a Gemini API key.
echo.

start "SignalGuard Backend" cmd /k "cd /d ""%~dp0"" && set ""PYTHONUTF8=1"" && set ""SIGNALGUARD_LIGHT_MODE=1"" && ""%PYTHON%"" -m api.app"
start "SignalGuard Frontend" cmd /k "cd /d ""%~dp0"" && ""%PYTHON%"" -m http.server 5500 --directory frontend"

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:5500/"

echo.
echo SignalGuard is starting.
echo Frontend: http://127.0.0.1:5500/
echo Backend:  http://127.0.0.1:5000/
echo.
echo Keep both command windows open while using SignalGuard.
echo Close those windows to stop SignalGuard.
echo.
pause
endlocal
