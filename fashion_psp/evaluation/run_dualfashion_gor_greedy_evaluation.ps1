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
$dualFashionRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $RepoRoot "fashion_psp\evaluation\evaluate_dualfashion_gor_greedy.py"
$output = Join-Path $dualFashionRoot "output\ifashion_gor_greedy_personal\evaluation"
$conda = Get-Command conda -ErrorAction Stop

$common = @(
    "--dualfashion-root", $dualFashionRoot,
    "--data-root", $DataRoot,
    "--processed-info-root", (Join-Path $dualFashionRoot "processed_info"),
    "--image-root", (Join-Path $DataRoot "semantic_category"),
    "--max-samples", "$MaxSamples",
    "--batch-size", "$BatchSize",
    "--device", $Device
)

foreach ($method in @("bon", "psp")) {
    Write-Host "[$method] evaluating final online greedy selections..."
    & $conda.Source run --no-capture-output --name $CondaEnvironment python $runner `
        --input-root (Join-Path $dualFashionRoot "output\ifashion_gor_greedy_personal\$method") `
        --output-root (Join-Path $output $method) `
        @common
    if ($LASTEXITCODE -ne 0) { throw "$method online greedy GOR evaluation failed." }
}
