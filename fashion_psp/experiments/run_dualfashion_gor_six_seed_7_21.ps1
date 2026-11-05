param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\dualfashion\python.exe",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$OutputRoot = "",
    [string]$OrderReferenceRoot = "",
    [int]$MaxSamples = 200,
    [int]$Seed = 2026,
    [string]$ResumeFrom = "auto",
    [string]$Device = "cuda:0",
    [switch]$SavePruneCandidates,
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$dualRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $dualRoot "run_ifashion_gor_greedy_personal.py"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $dualRoot "output\ifashion_gor_six_seed_7_21\personal_group0_seed$Seed"
}
if (-not $OrderReferenceRoot) {
    $OrderReferenceRoot = Join-Path $dualRoot "output\ifashion_gor\bon_offline_cache"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Runner not found: $runner" }
if (-not (Test-Path -LiteralPath $OrderReferenceRoot -PathType Container)) { throw "Missing order reference: $OrderReferenceRoot" }
$runArguments = @(
    $runner, "--method", "dsp",
    "--models-root", (Join-Path $dualRoot "checkpoint"), "--model-name", "0057501",
    "--data-root", $DataRoot, "--processed-info-root", (Join-Path $dualRoot "processed_info"),
    "--order-reference-root", $OrderReferenceRoot, "--output-root", $OutputRoot,
    "--max-samples", "$MaxSamples", "--seed", "$Seed",
    "--seed-stride", "10000", "--item-seed-stride", "64", "--num-inference-steps", "28",
    "--device", $Device, "--cpu-offload", "--reward-cpu-offload", "--resume-from", $ResumeFrom
)
if ($SavePruneCandidates) { $runArguments += "--save-prune-candidates" }
if ($PrepareOnly) { $runArguments += "--prepare-only" }
Write-Host "GOR fixed6: step7 keep4, step21 keep2, step28 select1; Personal only."
Write-Host "Budget: 112 particle-steps and 12 reward views per item; 448 and 48 per four-item outfit."
& $PythonExe @runArguments
if ($LASTEXITCODE -ne 0) { throw "GOR fixed6/7/21 failed; rerun with -ResumeFrom auto." }
