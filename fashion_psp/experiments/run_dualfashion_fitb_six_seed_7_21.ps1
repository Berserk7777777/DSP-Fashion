param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\dualfashion\python.exe",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$OutputRoot = "",
    [int]$MaxSamples = 2207,
    [int]$Seed = 2026,
    [string]$ResumeFrom = "auto",
    [string]$Device = "cuda:0",
    [string]$RewardDevice = "cuda:0",
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot "DualFashion-main\output\ifashion_fitb_six_seed_7_21\personal_group0_seed$Seed"
}
$runArguments = @(
    (Join-Path $PSScriptRoot "run_dualfashion_fitb_six_seed_7_21.py"),
    "--dualfashion-root", (Join-Path $RepoRoot "DualFashion-main"),
    "--data-root", $DataRoot,
    "--adaptive-policy", (Join-Path $RepoRoot "fashion_psp\artifacts\fixed_six_seed_7_21\frozen_rule.json"),
    "--output-root", $OutputRoot,
    "--max-samples", "$MaxSamples",
    "--seed", "$Seed",
    "--seed-stride", "10000",
    "--device", $Device,
    "--reward-device", $RewardDevice,
    "--cpu-offload",
    "--resume-from", $ResumeFrom
)
if ($PrepareOnly) { $runArguments += "--prepare-only" }
Write-Host "Python: $PythonExe"
Write-Host "Personal PSP@6: step7 keep4, step21 keep2, step28 keep1;112 particle steps."
& $PythonExe @runArguments
if ($LASTEXITCODE -ne 0) { throw "Fixed6/7/21 inference failed; rerun with -ResumeFrom auto." }
