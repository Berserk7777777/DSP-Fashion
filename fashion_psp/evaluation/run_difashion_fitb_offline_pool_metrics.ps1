param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\difashion\python.exe",
    [string]$CacheRoot = "",
    [string]$DataRoot = "D:\DiFashion\ifashion",
    [string]$ImageRoot = "",
    [string]$OutputRoot = "",
    [string]$Groups = "0,1,2",
    [string]$Methods = "bon,psp",
    [int]$MaxSamples = 2207,
    [int]$BatchSize = 4,
    [string]$Device = "cuda:0",
    [string]$PersonalClipWeights = "",
    [switch]$AuditOnly,
    [switch]$ExportOnly,
    [switch]$NoExcel,
    [switch]$RequireComplete
)
$ErrorActionPreference = "Stop"
if (-not $CacheRoot) { $CacheRoot = Join-Path $RepoRoot "DiFashion-main\offline_fitb_strict50" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $CacheRoot "evaluation" }
$arguments = @("-m", "fashion_psp.evaluation.evaluate_fitb_offline_pools", "--model", "difashion",
    "--cache-root", $CacheRoot, "--data-root", $DataRoot, "--output-root", $OutputRoot,
    "--groups", $Groups, "--methods", $Methods, "--max-tasks", "$MaxSamples",
    "--batch-size", "$BatchSize", "--device", $Device)
if ($ImageRoot) { $arguments += @("--image-root", $ImageRoot) }
if ($PersonalClipWeights) { $arguments += @("--personal-clip-weights", $PersonalClipWeights) }
if ($AuditOnly) { $arguments += "--audit-only" }
if ($ExportOnly) { $arguments += "--export-only" }
if ($NoExcel) { $arguments += "--no-excel" }
if ($RequireComplete) { $arguments += "--require-complete" }
Push-Location -LiteralPath $RepoRoot
try {
    & $PythonExe @arguments
    if ($LASTEXITCODE -ne 0) { throw "DiFashion offline metrics failed (exit $LASTEXITCODE)." }
} finally { Pop-Location }
