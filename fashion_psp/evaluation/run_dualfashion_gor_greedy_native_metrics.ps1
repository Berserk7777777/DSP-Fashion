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
$runner = Join-Path $RepoRoot "fashion_psp\evaluation\evaluate_dualfashion_gor_greedy_native_metrics.py"
$input = Join-Path $dual "output\ifashion_gor_greedy_personal"
$output = Join-Path $input "native_metrics"

foreach ($path in @($python, $runner, (Join-Path $input "bon"), (Join-Path $input "psp"))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}

$selections = @(
    "bon=$(Join-Path $input 'bon')",
    "psp=$(Join-Path $input 'psp')"
)
$selectionArgs = foreach ($entry in $selections) {
    "--selection"
    $entry
}

Write-Host "[GOR native metrics] frozen final selections; GT is post-selection for CIS and LPIPS."
& $python $runner @selectionArgs `
    --output-root $output `
    --dualfashion-root $dual `
    --data-root $DataRoot `
    --processed-info-root (Join-Path $dual "processed_info") `
    --image-root (Join-Path $DataRoot "semantic_category") `
    --max-samples $MaxSamples `
    --batch-size $BatchSize `
    --device $Device

if ($LASTEXITCODE -ne 0) { throw "Greedy GOR native metric evaluation failed." }
