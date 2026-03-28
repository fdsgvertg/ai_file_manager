@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: AI File Manager — Windows Explorer Right-Click Launcher
:: Drop this file in the project root.
:: Called automatically by the registry entry with the folder path as %1
:: ─────────────────────────────────────────────────────────────────────────────

setlocal EnableDelayedExpansion

:: ── Configuration ────────────────────────────────────────────────────────────
:: Set this to the absolute path of your project root
set "PROJECT_DIR=%~dp0"

:: Set this to your Python / conda environment
:: Option A: System Python
set "PYTHON_EXE=python"

:: Option B: Conda environment (uncomment and edit)
:: set "PYTHON_EXE=C:\Users\%USERNAME%\miniconda3\envs\ai_fm\python.exe"

:: Option C: venv (uncomment and edit)
:: set "PYTHON_EXE=%PROJECT_DIR%venv\Scripts\python.exe"

:: ── Target folder from right-click ──────────────────────────────────────────
set "TARGET_FOLDER=%~1"

if "%TARGET_FOLDER%"=="" (
    echo No folder specified. Right-click a folder and select "Sort using AI File Manager".
    pause
    exit /b 1
)

:: ── Mode selection dialog ────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║        AI FILE MANAGER — Sort Wizard         ║
echo  ╠══════════════════════════════════════════════╣
echo  ║  Folder: %TARGET_FOLDER%
echo  ╠══════════════════════════════════════════════╣
echo  ║  Select sorting mode:                        ║
echo  ║                                              ║
echo  ║    [1] By Topic     (AI content analysis)    ║
echo  ║    [2] By Date      (file modification date) ║
echo  ║    [3] By Relation  (semantic clustering)    ║
echo  ║    [4] Undo last sort                        ║
echo  ║    [0] Cancel                                ║
echo  ╚══════════════════════════════════════════════╝
echo.
set /p "CHOICE=Enter your choice (1-4): "

if "%CHOICE%"=="0" (
    echo Cancelled.
    exit /b 0
)

if "%CHOICE%"=="4" (
    echo.
    echo Running undo...
    cd /d "%PROJECT_DIR%"
    "%PYTHON_EXE%" cli.py undo "%TARGET_FOLDER%"
    echo.
    pause
    exit /b 0
)

if "%CHOICE%"=="1" set "MODE=topic"
if "%CHOICE%"=="2" set "MODE=date"
if "%CHOICE%"=="3" set "MODE=relation"

if not defined MODE (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

:: ── Preview first ────────────────────────────────────────────────────────────
echo.
echo  Running preview (no files will be moved yet)...
echo.

cd /d "%PROJECT_DIR%"
"%PYTHON_EXE%" cli.py sort "%TARGET_FOLDER%" --mode %MODE%

:: ── Confirm execution ────────────────────────────────────────────────────────
echo.
set /p "CONFIRM=Apply the sort shown above? [y/N]: "

if /i "%CONFIRM%"=="y" (
    echo.
    echo  Executing sort...
    "%PYTHON_EXE%" cli.py sort "%TARGET_FOLDER%" --mode %MODE% --execute --yes
) else (
    echo  Sort cancelled. No files were moved.
)

echo.
pause
