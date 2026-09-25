@echo off
rem One-click training for Windows: double-click this file.
rem Sets up .venv with the right PyTorch build, then downloads the data, builds graphs,
rem trains the encoder and draws the figures. Safe to re-run: every step resumes.
cd /d "%~dp0"
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 scripts\setup.py %*
) else (
    python scripts\setup.py %*
)
echo.
pause
