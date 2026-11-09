@echo off
REM webui.bat - launch the nano_SLMs Gradio web UI (webui/app.py).
REM Double-click, or run from any terminal. Extra args pass through, e.g.:
REM   webui.bat --no-browser --port 7861
REM   webui.bat --lan --auth user:pass --webhook https://example.com/hook
REM On any error the window STAYS OPEN and shows the full traceback.

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
    echo [error] .venv not found. Rebuild it first - see HANDOFF.md section 4.
    pause
    exit /b 1
)
if not exist "webui\app.py" (
    echo [error] webui\app.py not found.
    pause
    exit /b 1
)

echo Starting web UI on http://127.0.0.1:7860 ...
".venv\Scripts\python.exe" webui\app.py %*
set EXITCODE=%ERRORLEVEL%

echo.
if not "%EXITCODE%"=="0" (
    echo [error] web UI exited with code %EXITCODE%. Full error shown above.
) else (
    echo web UI exited normally.
)
pause
exit /b %EXITCODE%
