# install.ps1 — Automated installer for AI File Manager context menu
# Run as Administrator:  powershell -ExecutionPolicy Bypass -File install.ps1

param(
    [string]$InstallDir = "C:\ai_file_manager",
    [string]$PythonExe  = "python"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "  ╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "  ║     AI File Manager — Windows Installer      ║" -ForegroundColor Cyan
Write-Host "  ╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verify Python ─────────────────────────────────────────────────────────
Write-Host "[1/4] Checking Python..." -ForegroundColor Yellow
try {
    $pyVersion = & $PythonExe --version 2>&1
    Write-Host "      Found: $pyVersion" -ForegroundColor Green
} catch {
    Write-Error "Python not found at '$PythonExe'. Install Python 3.10+ and try again."
    exit 1
}

# ── 2. Create install directory ───────────────────────────────────────────────
Write-Host "[2/4] Setting up install directory: $InstallDir" -ForegroundColor Yellow
if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path $InstallDir | Out-Null
}

# Copy project files
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Write-Host "      Copying project files from $scriptDir..." -ForegroundColor Gray
Copy-Item -Path "$scriptDir\*" -Destination $InstallDir -Recurse -Force

# ── 3. Patch the .bat file with correct paths ─────────────────────────────────
Write-Host "[3/4] Configuring launcher..." -ForegroundColor Yellow
$batPath = "$InstallDir\windows\sort_files.bat"
$batContent = Get-Content $batPath -Raw
$batContent = $batContent -replace 'set "PROJECT_DIR=%~dp0"', "set `"PROJECT_DIR=$InstallDir\`""
$batContent = $batContent -replace 'set "PYTHON_EXE=python"', "set `"PYTHON_EXE=$PythonExe`""
Set-Content -Path $batPath -Value $batContent -Encoding ASCII
Write-Host "      Launcher configured ✓" -ForegroundColor Green

# Patch the .reg file with correct install directory
$regPath = "$InstallDir\windows\install_context_menu.reg"
$escapedDir = $InstallDir.Replace("\", "\\")
$regContent = Get-Content $regPath -Raw
$regContent = $regContent -replace "C:\\\\ai_file_manager", $escapedDir
$batEscaped  = $batPath.Replace("\", "\\")
$regContent = $regContent -replace "C:\\\\ai_file_manager\\\\windows\\\\sort_files.bat", $batEscaped
Set-Content -Path $regPath -Value $regContent -Encoding Unicode
Write-Host "      Registry script patched ✓" -ForegroundColor Green

# ── 4. Register context menu ──────────────────────────────────────────────────
Write-Host "[4/4] Installing Explorer context menu..." -ForegroundColor Yellow
Start-Process "regedit.exe" -ArgumentList "/s `"$regPath`"" -Wait -Verb RunAs
Write-Host "      Context menu registered ✓" -ForegroundColor Green

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ✅  Installation complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  Next steps:" -ForegroundColor Cyan
Write-Host "   1. Download model weights (see SETUP.md)" -ForegroundColor White
Write-Host "   2. Install Python deps:  pip install -r requirements.txt" -ForegroundColor White
Write-Host "   3. Right-click any folder in Explorer → 'Sort using AI File Manager'" -ForegroundColor White
Write-Host ""
