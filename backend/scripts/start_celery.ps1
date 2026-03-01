# 启动 Celery Worker 脚本
# 作用：设置环境变量并启动 Celery Worker (Windows兼容模式)

# 获取脚本所在目录的上级目录 (即 backend 目录)
$BackendRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $BackendRoot

Write-Host "Setting PYTHONPATH to: $env:PYTHONPATH" -ForegroundColor Cyan
Write-Host "Starting Celery Worker..." -ForegroundColor Green

# 检查 Redis 是否运行 (可选)
# Write-Host "Checking Redis status..."
# redis-cli ping

# 启动 Celery
# --pool=solo 是 Windows 必须的，因为不支持 prefork
# 必须在 backend 目录下运行，或者正确设置 PYTHONPATH
celery -A app.src.worker.celery_app worker --loglevel=info --pool=solo