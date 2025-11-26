# Quick GPU Status Check
Write-Host "`n================================================" -ForegroundColor Green
Write-Host "PyTorch GPU Detection" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
.venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "NVIDIA GPU Status" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
nvidia-smi --query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv

Write-Host "`n================================================" -ForegroundColor Green
Write-Host "Active GPU Processes" -ForegroundColor Green
Write-Host "================================================" -ForegroundColor Green
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
