@echo off
setlocal

cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8765"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"

if not exist "%PYTHON_EXE%" (
    echo LaunchPoint virtual environment was not found.
    echo Run install_dependencies.bat first, then run this file again.
    pause
    exit /b 1
)

echo Starting LaunchPoint at http://%HOST%:%PORT%
echo Press Ctrl+C in this window to stop the server.
echo.

"%PYTHON_EXE%" -m launchpoint.cli ui --host %HOST% --port %PORT%

if errorlevel 1 (
    echo.
    echo LaunchPoint stopped with an error.
    pause
)

