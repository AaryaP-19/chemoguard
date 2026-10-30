@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ===============================================
echo   ChemoGuard
echo ===============================================
echo.

REM ---------------------------------------------------------------
REM 1. Find a usable Python. The project is pinned to 3.12: Werkzeug
REM    3.x and scikit-learn 1.3.2 both ship wheels for it, and 3.13+
REM    has no scikit-learn 1.3.2 wheel.
REM ---------------------------------------------------------------
set "PYEXE="
py -3.12 --version >nul 2>&1 && set "PYEXE=py -3.12"
if not defined PYEXE (
    python --version 2>nul | findstr /c:"3.12" >nul && set "PYEXE=python"
)
if not defined PYEXE (
    echo [ERROR] Python 3.12 was not found.
    echo.
    echo The project is pinned to 3.12. Install it from python.org,
    echo tick "Add python.exe to PATH" during setup, then run this again.
    echo.
    pause
    exit /b 1
)
echo [ok] Found Python: %PYEXE%

REM ---------------------------------------------------------------
REM 2. Create the virtual environment if it is not already there.
REM ---------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [..] Creating virtual environment. This happens once.
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
    echo [ok] Virtual environment created.
)

set "VPY=.venv\Scripts\python.exe"

REM ---------------------------------------------------------------
REM 3. Install dependencies. The marker file means this only runs
REM    when requirements.txt is newer than the last install.
REM ---------------------------------------------------------------
set "NEEDINSTALL="
if not exist ".venv\.installed" set "NEEDINSTALL=1"
if exist ".venv\.installed" (
    for /f %%i in ('dir /b /o-d requirements.txt ".venv\.installed" 2^>nul') do (
        if "%%i"=="requirements.txt" set "NEEDINSTALL=1"
        goto :checked
    )
)
:checked

if defined NEEDINSTALL (
    echo [..] Installing dependencies. First run takes a few minutes.
    %VPY% -m pip install --upgrade pip --quiet
    %VPY% -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Dependency installation failed. See the messages above.
        pause
        exit /b 1
    )
    echo installed > ".venv\.installed"
    echo [ok] Dependencies installed.
) else (
    echo [ok] Dependencies already installed.
)

REM ---------------------------------------------------------------
REM 4. Configuration file.
REM ---------------------------------------------------------------
if not exist ".env" (
    echo.
    echo [WARN] No .env file found.
    if exist "env.example" (
        copy /y "env.example" ".env" >nul
        echo        A blank .env has been created from env.example.
        echo        Open it and fill in DATABASE_URL and, if you want the
        echo        vision backend, OPENAI_API_KEY. Leaving DATABASE_URL
        echo        empty falls back to a local SQLite file.
    )
    echo.
)

REM ---------------------------------------------------------------
REM 5. Tesseract is an external program and cannot be bundled.
REM ---------------------------------------------------------------
where tesseract >nul 2>&1
if errorlevel 1 (
    if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
        set "PATH=C:\Program Files\Tesseract-OCR;%PATH%"
        echo [ok] Tesseract found, added to PATH for this session.
    ) else (
        echo [WARN] Tesseract not found. OCR extraction will not work.
        echo        Install from: https://github.com/UB-Mannheim/tesseract/wiki
        echo        The rest of the application still runs.
    )
) else (
    echo [ok] Tesseract on PATH.
)

REM ---------------------------------------------------------------
REM 6. Start.
REM ---------------------------------------------------------------
echo.
echo ===============================================
echo   Starting on http://localhost:5000
echo   Press Ctrl+C in this window to stop.
echo ===============================================
echo.

start "" "http://localhost:5000"
%VPY% app.py

echo.
echo Server stopped.
pause
