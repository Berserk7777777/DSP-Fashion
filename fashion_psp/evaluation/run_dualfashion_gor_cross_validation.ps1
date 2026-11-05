param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$MaxSamples = 200,
    [int]$BatchSize = 8,
    [string]$CondaEnvironment = "dualfashion",
    [string]$PCARCheckpoint = "E:\difashion-PSP\fashion_psp\artifacts\pcar\ifashion_train_v2\best.pt",
    [switch]$SkipGtMetrics
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$dualFashionRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $RepoRoot "fashion_psp\evaluation\evaluate_dualfashion_gor_offline.py"
$processedInfo = Join-Path $dualFashionRoot "processed_info"
$imageRoot = Join-Path $DataRoot "semantic_category"
$evaluationRoot = Join-Path $dualFashionRoot "output\ifashion_gor\evaluation"

if (-not (Test-Path -LiteralPath $runner)) { throw "Evaluator not found: $runner" }
$conda = Get-Command conda -ErrorAction Stop

$common = @(
    "--dualfashion-root", $dualFashionRoot,
    "--data-root", $DataRoot,
    "--processed-info-root", $processedInfo,
    "--image-root", $imageRoot,
    "--max-samples", "$MaxSamples",
    "--batch-size", "$BatchSize",
    "--device", $Device,
    "--pcar-checkpoint", $PCARCheckpoint,
    "--resume-from", "auto"
)
if (-not $SkipGtMetrics) { $common += "--with-gt-metrics" }

$runs = @(
    @{ Name = "BoN"; Cache = (Join-Path $dualFashionRoot "output\ifashion_gor\bon_offline_cache"); Output = (Join-Path $evaluationRoot "bon") },
    @{ Name = "PSP"; Cache = (Join-Path $dualFashionRoot "output\ifashion_gor\psp_offline_cache"); Output = (Join-Path $evaluationRoot "psp") }
)

foreach ($run in $runs) {
    Write-Host "[$($run.Name)] evaluating Compatibility, Personal, Bottleneck, and PCAR..."
    & $conda.Source run --no-capture-output --name $CondaEnvironment python $runner `
        --cache-root $run.Cache `
        --output-root $run.Output `
        @common
    if ($LASTEXITCODE -ne 0) { throw "$($run.Name) GOR evaluation failed." }
}
