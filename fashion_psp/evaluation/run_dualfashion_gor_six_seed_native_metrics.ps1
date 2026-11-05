param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\dualfashion\python.exe",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$SelectionRoot = "",
    [string]$OutputRoot = "",
    [string]$Device = "cuda:0",
    [int]$MaxSamples = 200,
    [int]$BatchSize = 4
)
$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$dualRoot = Join-Path $RepoRoot "DualFashion-main"
if (-not $SelectionRoot) {
    $SelectionRoot = Join-Path $dualRoot "output\ifashion_gor_six_seed_7_21\personal_group0_seed2026"
}
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $dualRoot "output\ifashion_gor_six_seed_7_21\native_metrics"
}
$runner = Join-Path $PSScriptRoot "evaluate_dualfashion_gor_greedy_native_metrics.py"
foreach ($path in @($PythonExe, $runner, $SelectionRoot)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
Write-Host "GOR six-seed 7/21: evaluate frozen final images only."
& $PythonExe $runner `
    --selection "dsp_six_seed_7_21=$SelectionRoot" `
    --output-root $OutputRoot `
    --dualfashion-root $dualRoot `
    --data-root $DataRoot `
    --processed-info-root (Join-Path $dualRoot "processed_info") `
    --image-root (Join-Path $DataRoot "semantic_category") `
    --max-samples $MaxSamples `
    --batch-size $BatchSize `
    --device $Device
if ($LASTEXITCODE -ne 0) { throw "GOR six-seed native metric evaluation failed." }
