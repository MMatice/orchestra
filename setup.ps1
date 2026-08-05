# Installation d'Orchestra : venv isole, verification d'Ollama, telechargement
# des modeles correspondant au profil materiel detecte.
#
#   .\setup.ps1              installe tout
#   .\setup.ps1 -SkipModels  installe seulement les dependances Python

param(
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$venvPython = Join-Path $root ".venv\Scripts\python.exe"

Write-Host "== Orchestra : installation ==" -ForegroundColor Cyan

# --- Environnement Python isole ---------------------------------------------
# Un venv dedie evite que les dependances du serveur MCP (starlette, httpx...)
# entrent en conflit avec le Python systeme.
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $python) {
    Write-Host "Python introuvable dans le PATH." -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $venvPython)) {
    Write-Host "`n-- Creation du venv --" -ForegroundColor Cyan
    & $python -m venv (Join-Path $root ".venv")
    if ($LASTEXITCODE -ne 0) { Write-Host "Echec." -ForegroundColor Red; exit 1 }
}
Write-Host "Interpreteur : $venvPython"

Write-Host "`n-- Dependances --" -ForegroundColor Cyan
& $venvPython -m pip install --quiet --disable-pip-version-check --upgrade pip
& $venvPython -m pip install --quiet --disable-pip-version-check -r (Join-Path $root "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    Write-Host "Echec de l'installation des dependances." -ForegroundColor Red
    exit 1
}
Write-Host "OK"

# --- Ollama -----------------------------------------------------------------
Write-Host "`n-- Serveur Ollama --" -ForegroundColor Cyan
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    Write-Host "Ollama n'est pas installe. https://ollama.com/download" -ForegroundColor Red
    exit 1
}

function Test-Ollama {
    try {
        Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/version" -TimeoutSec 3 | Out-Null
        return $true
    } catch {
        return $false
    }
}

if (-not (Test-Ollama)) {
    Write-Host "Ollama ne repond pas, demarrage en arriere-plan..."
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    # Le serveur met quelques secondes a ouvrir son port au premier lancement.
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Test-Ollama) { break }
    }
}

if (Test-Ollama) {
    Write-Host "OK"
} else {
    Write-Host "Ollama n'a pas demarre. Lance 'ollama serve' manuellement." -ForegroundColor Yellow
}

# --- Profil + modeles -------------------------------------------------------
Push-Location $root
try {
    Write-Host "`n-- Profil materiel detecte --" -ForegroundColor Cyan
    & $venvPython -m orchestra.cli status

    if (-not $SkipModels) {
        Write-Host "`n-- Telechargement des modeles du profil --" -ForegroundColor Cyan
        Write-Host "(plusieurs Go, la premiere fois c'est long)"
        & $venvPython -m orchestra.cli pull
    }
} finally {
    Pop-Location
}

# --- Enregistrement MCP -----------------------------------------------------
Write-Host "`n== Etape finale ==" -ForegroundColor Cyan
Write-Host "Enregistre le serveur MCP dans Claude Code :`n"
Write-Host "  claude mcp add orchestra --scope user -- `"$venvPython`" -m orchestra.mcp_server" -ForegroundColor Green
Write-Host "`nLance la commande depuis $root (le serveur resout ses agents depuis ce dossier)."
