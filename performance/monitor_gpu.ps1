# GPU Monitoring Script
# Run this in a separate terminal while your performance test runs

Write-Host "GPU Monitoring - Press Ctrl+C to stop" -ForegroundColor Green
Write-Host "Refreshing every 2 seconds...`n" -ForegroundColor Yellow

while ($true) {
    Clear-Host
    Write-Host "==================================================" -ForegroundColor Cyan
    Write-Host "GPU Status - $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Cyan
    Write-Host "==================================================" -ForegroundColor Cyan
    
    # Run nvidia-smi to show GPU usage
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
    
    Write-Host "`n--------------------------------------------------" -ForegroundColor Cyan
    Write-Host "Detailed Process Info:" -ForegroundColor Cyan
    Write-Host "--------------------------------------------------" -ForegroundColor Cyan
    
    # Show processes using GPU
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
    
    Start-Sleep -Seconds 2
}
