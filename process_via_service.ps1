param(
    [string]$InputPath = $PSScriptRoot,
    [string]$OutputPath = (Join-Path $PSScriptRoot "outputs_service"),
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$body = @{
    input = $InputPath
    output = $OutputPath
    force = [bool]$Force
} | ConvertTo-Json -Compress

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8765/process" `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
