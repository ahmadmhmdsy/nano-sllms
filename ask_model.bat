@echo off
REM ask_model.bat - interactive prompting REPL for the trained nano_SLMs model.
REM Double-click, or run from any terminal. Extra args pass through to infer.py,
REM e.g.: ask_model.bat --sample --temperature 0.8 --max_new_tokens 128 --prompt "def f():"
REM Type a code prefix and press Enter inside the REPL; empty line or Ctrl+Z+Enter quits.

cd /d "%~dp0"
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

if not exist ".venv\Scripts\python.exe" (
    echo [error] .venv not found. Rebuild it first - see HANDOFF.md section 4.
    pause
    exit /b 1
)
if not exist "runs\pilot\final" (
    echo [error] runs\pilot\final not found - nothing trained yet. Train first:
    echo     .venv\Scripts\python.exe scripts\train.py --config configs\pilot.yaml
    pause
    exit /b 1
)

".venv\Scripts\python.exe" scripts\infer.py --config configs\pilot.yaml %*
pause
