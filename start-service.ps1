param([int]$Port = 8000)
& (Join-Path $PSScriptRoot 'venv\Scripts\python.exe') -B (Join-Path $PSScriptRoot 'run.py') --config (Join-Path $PSScriptRoot 'service.local.json') --host 127.0.0.1 --port $Port
exit $LASTEXITCODE

