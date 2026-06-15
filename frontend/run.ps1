# Frontend dev server (Webpack mode — more stable on OneDrive-synced folders than Turbopack).
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Stopping stale frontend processes on port 3000..."
$procIds = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $procIds) {
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

if (Test-Path ".next") {
    Write-Host "Clearing .next cache..."
    Remove-Item -Recurse -Force ".next" -ErrorAction SilentlyContinue
}

Write-Host "Starting frontend on http://localhost:3000 (Webpack dev server)"
npm run dev
