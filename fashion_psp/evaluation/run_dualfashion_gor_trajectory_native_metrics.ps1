param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$MaxSamples = 200,
    [int]$BatchSize = 4,
    [string]$CondaEnvironment = "dualfashion"
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$python = "D:\anaconda\envs\$CondaEnvironment\python.exe"
$dual = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $RepoRoot "fashion_psp\evaluation\evaluate_dualfashion_gor_trajectory_native_metrics.py"
$tasks = Join-Path $dual "output\ifashion_gor\evaluation\bon\tasks\personal"
$output = Join-Path $dual "output\ifashion_gor\native_metrics_personal"

foreach ($path in @($python, $runner, $tasks)) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}

Write-Host "[Trajectory GOR native metrics] frozen Standard and Personal-BoN selections; GT is post-selection only."
& $python $runner `
    --tasks-root $tasks `
    --output-root $output `
    --dualfashion-root $dual `
    --data-root $DataRoot `
    --processed-info-root (Join-Path $dual "processed_info") `
    --image-root (Join-Path $DataRoot "semantic_category") `
    --max-samples $MaxSamples `
    --batch-size $BatchSize `
    --device $Device

if ($LASTEXITCODE -ne 0) { throw "Trajectory GOR native metric evaluation failed." }
