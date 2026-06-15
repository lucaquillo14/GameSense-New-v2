# Always use Python 3.12 — default `python` on this machine is 3.14 and cannot run CV deps.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Installing requirements (Python 3.12)..."
py -3.12 -m pip install -r requirements.txt -q

Write-Host "Stopping stale backend processes on port 8000..."
$procIds = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique
foreach ($procId in $procIds) {
    Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 2

Write-Host "Starting backend on http://127.0.0.1:8000"
py -3.12 -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000 --limit-max-requests 1000 --timeout-keep-alive 120
