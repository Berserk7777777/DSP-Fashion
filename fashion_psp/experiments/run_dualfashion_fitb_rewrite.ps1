param(
    [string]$Python = "",
    [string]$OutputDir = "",
    [ValidateSet("order", "history")][string]$RewriteMode = "order",
    [switch]$PrepareOnly,
    [switch]$AuditAllRewrites,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Python) {
    if ($env:CONDA_PREFIX -and (Test-Path -LiteralPath (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    } else {
        throw "Activate conda environment dualfashion first, or pass -Python with its python.exe."
    }
}
& $Python -c "import sys, torch, open_clip, diffusers; print('Python:', sys.executable); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Python environment is missing DualFashion dependencies." }
$env:HF_HUB_OFFLINE = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$arguments = @(
    (Join-Path $PSScriptRoot "run_dualfashion_fitb_rewrite.py"),
    "--dualfashion-root", (Join-Path $repoRoot "DualFashion-main"),
    "--task-index", "395", "--seed", "2026", "--cpu-offload",
    "--rewrite-mode", $RewriteMode,
    "--device", "cuda:0", "--reward-device", "cpu"
)
if (-not $OutputDir -and $RewriteMode -eq "history") {
    $OutputDir = Join-Path $repoRoot "DualFashion-main\output\ifashion_fitb_rewrite\task395_9674_18277_history_modes"
}
if ($OutputDir) { $arguments += @("--output-dir", $OutputDir) }
if ($PrepareOnly) { $arguments += "--prepare-only" }
if ($AuditAllRewrites) { $arguments += "--audit-all-rewrites" }
if ($Resume) { $arguments += "--resume" }
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Rewrite-PSP failed; rerun the same configuration with -Resume." }
