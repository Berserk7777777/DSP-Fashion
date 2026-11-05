param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$PythonExe = "",
    [string]$CondaEnvironment = "dualfashion",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 4,
    [switch]$SkipSelection
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
if (-not $PythonExe) {
    if ($env:CONDA_PREFIX -and (Split-Path -Leaf $env:CONDA_PREFIX) -eq $CondaEnvironment) {
        $PythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
    } else {
        $conda = Get-Command conda -ErrorAction Stop
        $environmentJson = & $conda.Source env list --json
        if ($LASTEXITCODE -ne 0) { throw "Cannot list conda environments." }
        $environments = $environmentJson | ConvertFrom-Json
        $prefix = @($environments.envs | Where-Object { (Split-Path -Leaf $_) -eq $CondaEnvironment })
        if ($prefix.Count -ne 1) { throw "Pass -PythonExe with the full dualfashion python.exe path." }
        $PythonExe = Join-Path $prefix[0] "python.exe"
    }
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
Write-Host "Python: $PythonExe"
& $PythonExe -c "import sys, torch, open_clip, lpips; print(sys.executable); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Missing evaluation dependencies in this interpreter." }

$evaluationRoot = Join-Path $RepoRoot "fashion_psp\evaluation"
$outputRoot = Join-Path $RepoRoot "DualFashion-main\output\ifashion_fitb_state_risk_dp"
if (-not $SkipSelection) {
    & $PythonExe (Join-Path $evaluationRoot "research_dualfashion_state_risk_dp.py") --output-dir $outputRoot
    if ($LASTEXITCODE -ne 0) { throw "State-conditioned DP replay failed." }
}
& $PythonExe (Join-Path $evaluationRoot "evaluate_dualfashion_state_risk_dp.py") `
    --selection-csv (Join-Path $outputRoot "frozen_selections.csv") `
    --output-dir (Join-Path $outputRoot "native_metrics") `
    --dualfashion-root (Join-Path $RepoRoot "DualFashion-main") `
    --data-root $DataRoot --device $Device --batch-size $BatchSize --resume
if ($LASTEXITCODE -ne 0) { throw "Frozen selection metrics failed; rerun with -SkipSelection to resume." }
Write-Host "Summary: $(Join-Path $outputRoot 'native_metrics\summary.csv')"
