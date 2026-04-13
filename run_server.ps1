$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:PYTHONPATH = (Join-Path $root 'deps')
$env:SENTENCE_TRANSFORMERS_HOME = (Join-Path $root 'models')
$env:HF_HOME = (Join-Path $root 'models\.cache')
$env:TRANSFORMERS_CACHE = (Join-Path $root 'models\.cache')
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

Set-Location $root
python -m uvicorn main:app --host 0.0.0.0 --port 8000
