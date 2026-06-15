# One-shot fix: kill every stale backend, verify this folder has the v2 code,
# start the backend (port 8000, or 8010 if 8000 cannot be freed), point the
# frontend at the right port, and confirm /health reports the v2 engine.
# TIP: run this in an *Administrator* PowerShell for best results.
$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot

Write-Host "== Step 1: killing stale backends on ports 8000/8001 and stray uvicorn ==" -ForegroundColor Cyan
$pids = @()
foreach ($port in 8000, 8001) {
    $pids += Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess
}
$pids += (Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*uvicorn*" }).ProcessId
$pids = $pids | Where-Object { $_ -and $_ -gt 4 } | Sort-Object -Unique
foreach ($procId in $pids) {
    $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host ("  killing PID {0} ({1}) and its children" -f $procId, $proc.ProcessName)
        & taskkill /PID $procId /T /F 2>$null | Out-Null
    }
}
Start-Sleep -Seconds 2

# Diagnose anything still on 8000 and pick the port to use
$backendPort = 8000
$holder = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($holder) {
    $hp = Get-Process -Id $holder.OwningProcess -ErrorAction SilentlyContinue
    Write-Host ("WARNING: port 8000 is still held by PID {0} ({1})" -f $holder.OwningProcess, $hp.ProcessName) -ForegroundColor Yellow
    if ($hp -and $hp.Path) { Write-Host ("  path: " + $hp.Path) -ForegroundColor Yellow }
    & taskkill /PID $holder.OwningProcess /T /F 2>$null | Out-Null
    Start-Sleep -Seconds 2
    if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
        Write-Host "  could not free port 8000 - falling back to port 8010" -ForegroundColor Yellow
        $backendPort = 8010
        $stale8010 = Get-NetTCPConnection -LocalPort 8010 -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if ($stale8010) { & taskkill /PID $stale8010.OwningProcess /T /F 2>$null | Out-Null; Start-Sleep -Seconds 1 }
    }
}
Write-Host ("  using backend port: " + $backendPort) -ForegroundColor Green

Write-Host "== Step 2: verifying this folder contains the v2 code ==" -ForegroundColor Cyan
if (-not (Select-String -Path "app\main.py" -Pattern "v2-parallel" -Quiet)) {
    Write-Host "ERROR: app\main.py in this folder is NOT the v2 code (wrong folder?)" -ForegroundColor Red
    Write-Host ("This script ran from: " + $PSScriptRoot)
    exit 1
}
if (-not (Select-String -Path "app\services\shooting_technique_pipeline.py" -Pattern "_detect_batch" -Quiet)) {
    Write-Host "ERROR: pipeline file is missing the v2 parallel code." -ForegroundColor Red
    exit 1
}
Write-Host "  v2 markers found - this folder is up to date." -ForegroundColor Green

Write-Host "== Step 3: pointing the frontend at the backend port ==" -ForegroundColor Cyan
$envFile = Join-Path (Split-Path $PSScriptRoot -Parent) "frontend\.env.local"
Set-Content -Path $envFile -Value ("NEXT_PUBLIC_API_URL=http://localhost:" + $backendPort) -Encoding Ascii
Write-Host ("  wrote " + $envFile)

Write-Host "== Step 4: installing requirements (Python 3.12) ==" -ForegroundColor Cyan
py -3.12 -m pip install -r requirements.txt -q

Write-Host ("== Step 5: starting backend on http://127.0.0.1:" + $backendPort + " ==") -ForegroundColor Cyan
$server = Start-Process -FilePath "py" `
    -ArgumentList "-3.12", "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "$backendPort" `
    -WorkingDirectory $PSScriptRoot -PassThru

Write-Host "== Step 6: waiting for /health to report the v2 engine ==" -ForegroundColor Cyan
$ok = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 1
    try {
        $h = Invoke-RestMethod ("http://127.0.0.1:" + $backendPort + "/health") -TimeoutSec 2
        if ($h.shooting_technique_engine -eq "rfdetr-mediapipe-v2-parallel") { $ok = $true; break }
        Write-Host ("  health reports engine: '{0}' - not v2 yet..." -f $h.shooting_technique_engine)
    } catch { }
}
if ($ok) {
    Write-Host ""
    Write-Host ("SUCCESS: v2 backend is live on http://127.0.0.1:" + $backendPort + " (PID " + $server.Id + ").") -ForegroundColor Green
    Write-Host "Leave it running. Now RESTART the frontend: Ctrl+C in its window, then: npm run dev"
    Write-Host "Then upload your clip again."
} else {
    Write-Host ""
    Write-Host "FAILED: backend did not report the v2 engine within 30s." -ForegroundColor Red
    Write-Host "Copy ALL output from this window and paste it to Claude."
}
