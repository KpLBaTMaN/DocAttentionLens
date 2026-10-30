param([string]$Python = "")
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
if (-not $Python) {
    if ($env:DOCATTENTIONLENS_PYTHON) { $Python = $env:DOCATTENTIONLENS_PYTHON }
    elseif (Get-Command python -ErrorAction SilentlyContinue) { $Python = 'python' }
    else { throw 'Python 3.12 is required. Pass -Python with its executable path.' }
}
& $Python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 'Python 3.12 is required.')"
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 is required.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.lock
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e library -e backend
if ($LASTEXITCODE -ne 0) { throw 'Local package installation failed.' }
npm ci --prefix frontend
if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed.' }
npm run api:generate --prefix frontend
if ($LASTEXITCODE -ne 0) { throw 'API generation failed.' }
Write-Output 'Environment ready. See README.md for startup and verification.'
