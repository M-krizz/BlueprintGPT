param(
    [ValidateSet('learned', 'algorithmic', 'all')]
    [string]$Mode = 'all'
)

$ErrorActionPreference = 'Stop'
$python = "d:/Projects/GenAI/.venv/Scripts/python.exe"

function Invoke-Demo([string]$Label, [string]$ModuleName) {
    Write-Host "== $Label =="
    & $python -m $ModuleName
    if ($LASTEXITCODE -ne 0) {
        throw "$Label demo failed with exit code $LASTEXITCODE"
    }
    Write-Host ""
}

if ($Mode -in @('learned', 'all')) {
    Invoke-Demo 'Learned demo' 'learned.integration.run_smoke_learned'
}

if ($Mode -in @('algorithmic', 'all')) {
    Invoke-Demo 'Algorithmic baseline demo' 'demo.run_smoke_algorithmic'
}
