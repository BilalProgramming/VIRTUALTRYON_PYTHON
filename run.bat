@echo off
call venv\Scripts\activate
python main.py
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Application failed to start.
    echo Please ensure you have installed "Visual C++ Redistributable 2015-2022".
    echo Download: https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    pause
)
pause
