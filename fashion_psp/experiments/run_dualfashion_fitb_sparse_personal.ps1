param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\dualfashion\python.exe",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$OutputRoot = "",
    [string]$AdaptivePolicyPath = "",
    [int]$MaxSamples = 2207,
    [int]$Seed = 2026,
    [string]$ResumeFrom = "auto",
    [string]$Device = "cuda:0",
    [string]$RewardDevice = "cuda:0",
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot "DualFashion-main\output\ifashion_fitb_sparse_personal\personal_group0_seed$Seed"
}
if (-not $AdaptivePolicyPath) {
    $AdaptivePolicyPath = Join-Path $RepoRoot "fashion_psp\artifacts\sparse_personal_development_v3\selected_policy.json"
}
$runArguments = @(
    (Join-Path $PSScriptRoot "run_dualfashion_fitb_sparse_personal.py"),
    "--dualfashion-root", (Join-Path $RepoRoot "DualFashion-main"),
    "--data-root", $DataRoot,
    "--adaptive-policy", $AdaptivePolicyPath,
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
& $PythonExe @runArguments
if ($LASTEXITCODE -ne 0) { throw "Sparse Personal PSP failed; rerun the same command with -ResumeFrom auto." }
