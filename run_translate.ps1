$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Python virtual environment not found: $python"
}

& $python (Join-Path $PSScriptRoot "translate_manga.py") `
    --input $PSScriptRoot `
    --output (Join-Path $PSScriptRoot "outputs")
