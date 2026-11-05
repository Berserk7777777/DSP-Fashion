param(
    [ValidateSet("fitb", "gor", "both")]
    [string]$Task = "both",
    [string]$TaskIndices = "0,1,2,3,4,5,6,7,8,9",
    [int]$Warmup = 1,
    [ValidateSet("personal", "none")]
    [string]$Reward = "personal",
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $env:CONDA_PREFIX "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Activate the dualfashion conda environment before running this script. CONDA_PREFIX=$env:CONDA_PREFIX"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$output = Join-Path $root "fashion_psp\benchmarks\results\dualfashion_parallel_$Task`_$Reward`_$stamp"
& $python (Join-Path $root "fashion_psp\benchmarks\benchmark_dualfashion_parallel_search.py") `
    --task $Task `
    --models-root (Join-Path $root "DualFashion-main\checkpoint") `
    --data-root "C:\tmp\ifashion" `
    --processed-info-root (Join-Path $root "DualFashion-main\processed_info") `
    --image-root "C:\tmp\ifashion\semantic_category" `
    --task-indices $TaskIndices `
    --warmup $Warmup `
    --reward $Reward `
    --device $Device `
    --output-dir $output
if ($LASTEXITCODE -ne 0) { throw "Parallel benchmark failed." }

Write-Host "Results: $output\summary.csv"
