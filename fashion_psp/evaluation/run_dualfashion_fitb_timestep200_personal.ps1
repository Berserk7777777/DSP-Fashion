param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$MaxSamples = 200,
    [int]$BatchSize = 4,
    [string]$CondaEnvironment = "dualfashion",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$dualFashionRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $RepoRoot "fashion_psp\evaluation\evaluate_dualfashion_fitb_timestep.py"
$cacheRoot = Join-Path $dualFashionRoot "output\ifashion_fitb_timestep200_cache"
$outputRoot = Join-Path $dualFashionRoot "output\ifashion_fitb_timestep200_personal_analysis"

if (-not (Test-Path -LiteralPath $runner)) { throw "Evaluator not found: $runner" }
if (-not (Test-Path -LiteralPath $cacheRoot)) { throw "Cache not found: $cacheRoot" }
$conda = Get-Command conda -ErrorAction Stop

$arguments = @(
    $runner,
    "--cache-root", $cacheRoot,
    "--output-root", $outputRoot,
    "--dualfashion-root", $dualFashionRoot,
    "--data-root", $DataRoot,
    "--processed-info-root", (Join-Path $dualFashionRoot "processed_info"),
    "--image-root", (Join-Path $DataRoot "semantic_category"),
    "--max-samples", "$MaxSamples",
    "--batch-size", "$BatchSize",
    "--device", $Device
)
if ($Resume) { $arguments += "--resume" }

& $conda.Source run --no-capture-output --name $CondaEnvironment python @arguments
if ($LASTEXITCODE -ne 0) { throw "FITB timestep Personal evaluation failed." }
