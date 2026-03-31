$ErrorActionPreference = "Stop"

$serviceUrl = "http://127.0.0.1:8765/health"
$serviceScript = Join-Path $PSScriptRoot "run_service.ps1"
$processScript = Join-Path $PSScriptRoot "process_via_service.ps1"

function Test-ServiceReady {
    try {
        Invoke-RestMethod -Uri $serviceUrl -Method Get -TimeoutSec 2 | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

if (-not (Test-ServiceReady)) {
    Start-Process -FilePath "powershell.exe" -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $serviceScript
    ) -WorkingDirectory $PSScriptRoot | Out-Null

    $ready = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep -Seconds 2
        if (Test-ServiceReady) {
            $ready = $true
            break
        }
    }

    if (-not $ready) {
        throw "Service did not become ready in time."
    }
}

& $processScript
