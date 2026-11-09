@echo off
REM diacritize.bat - interactive Arabic diacritization chat with the D-line GOLD
REM model (stage2b2500, 30M, best-gate snapshot at step 2500).
REM Double-click, or run from any terminal. Extra args pass through:
REM   diacritize.bat --file my_text.txt
REM   diacritize.bat --text "<arabic sentence>"
REM In interactive mode: type/paste Arabic -> diacritized output; q quits.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [error] .venv not found - rebuild it first, see ENVIRONMENT.md.
    pause
    exit /b 1
)
if not exist "runs\diac\stage2b2500\final\model.pt" (
    echo [error] GOLD model missing: runs\diac\stage2b2500\final\model.pt
    echo         see HANDOFF.md / research/FINAL_REPORT.md
    pause
    exit /b 1
)

echo [diacritize] loading the GOLD model... first answer needs a few seconds.
".venv\Scripts\python.exe" -X utf8 diacritizer\scripts\diacritize.py %*
echo.
echo [diacritize] session ended
pause
