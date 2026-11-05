param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$MaxSamples = 200,
    [string]$CondaEnvironment = "dualfashion",
    [switch]$CpuOffload,
    [switch]$SavePruneCandidates,
    [string]$ResumeFrom = "auto"
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

$dualFashionRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $dualFashionRoot "run_ifashion_gor_greedy_personal.py"
$orderReference = Join-Path $dualFashionRoot "output\ifashion_gor\bon_offline_cache"
$outputBase = Join-Path $dualFashionRoot "output\ifashion_gor_greedy_personal"
$conda = Get-Command conda -ErrorAction Stop

if (-not (Test-Path -LiteralPath $runner)) { throw "Runner not found: $runner" }
if (-not (Test-Path -LiteralPath $orderReference)) { throw "BoN standard order cache not found: $orderReference" }

$common = @(
    "--models-root", (Join-Path $dualFashionRoot "checkpoint"),
    "--model-name", "0057501",
    "--data-root", $DataRoot,
    "--processed-info-root", (Join-Path $dualFashionRoot "processed_info"),
    "--order-reference-root", $orderReference,
    "--max-samples", "$MaxSamples",
    "--num-inference-steps", "28",
    "--device", $Device,
    "--resume-from", $ResumeFrom
)
if ($CpuOffload) { $common += "--cpu-offload" }
if ($CpuOffload) { $common += "--reward-cpu-offload" }
if ($SavePruneCandidates) { $common += "--save-prune-candidates" }

$runs = @(
    @{ Name = "greedy-BoN"; Method = "bon"; Seed = "42"; Output = (Join-Path $outputBase "bon") },
    @{ Name = "greedy-PSP"; Method = "psp"; Seed = "2026"; Output = (Join-Path $outputBase "psp") }
)
foreach ($run in $runs) {
    Write-Host "[$($run.Name)] fixed BoN-standard order; Personal reward; item-level online selection"
    & $conda.Source run --no-capture-output --name $CondaEnvironment python $runner `
        "--method" $run.Method `
        "--seed" $run.Seed `
        "--output-root" $run.Output `
        @common
    if ($LASTEXITCODE -ne 0) { throw "$($run.Name) inference failed." }
}
