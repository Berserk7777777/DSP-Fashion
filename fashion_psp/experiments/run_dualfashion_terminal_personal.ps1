param(
    [string]$Python = "",
    [string]$MetricsCsv = "",
    [string]$OutputDir = "",
    [int]$MaxTasks = 200,
    [int]$EnsembleSize = 5,
    [int]$HiddenDim = 64,
    [int]$Epochs = 100,
    [int]$Patience = 15,
    [int]$BatchSize = 1024,
    [double]$LearningRate = 0.001,
    [double]$Coverage = 0.90,
    [int]$Seed = 2026,
    [string]$Device = "cuda:0",
    [switch]$Resume,
    [switch]$ReplayOnly,
    [switch]$PosthocGT
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $Python) {
    if ($env:CONDA_PREFIX -and (Test-Path -LiteralPath (Join-Path $env:CONDA_PREFIX "python.exe"))) {
        $Python = Join-Path $env:CONDA_PREFIX "python.exe"
    } else {
        throw "Activate dualfashion first, or supply -Python with its python.exe."
    }
}
if (-not $MetricsCsv) {
    $MetricsCsv = Join-Path $repoRoot "DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis\candidate_metrics_pre_gt.csv"
}
if (-not $OutputDir) {
    $OutputDir = Join-Path $repoRoot "fashion_psp\artifacts\terminal_personal\fitb200_v1"
}
if (-not (Test-Path -LiteralPath $MetricsCsv)) { throw "Personal trajectory scores not found: $MetricsCsv" }
& $Python -c "import sys, torch, numpy, tqdm; print('Python:', sys.executable); print('Torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { throw "Missing training dependencies in selected Python." }
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$arguments = @(
    (Join-Path $PSScriptRoot "train_dualfashion_terminal_personal.py"),
    "--metrics-csv", $MetricsCsv, "--output-dir", $OutputDir,
    "--max-tasks", "$MaxTasks", "--ensemble-size", "$EnsembleSize",
    "--hidden-dim", "$HiddenDim", "--epochs", "$Epochs", "--patience", "$Patience",
    "--batch-size", "$BatchSize", "--learning-rate", "$LearningRate",
    "--coverage", "$Coverage", "--seed", "$Seed", "--device", $Device
)
if ($Resume) { $arguments += "--resume" }
if ($ReplayOnly) { $arguments += "--replay-only" }
if ($PosthocGT) {
    $posthoc = Join-Path $repoRoot "DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis\candidate_metrics_with_gt.csv"
    if (-not (Test-Path -LiteralPath $posthoc)) { throw "Post-hoc diagnostics not found: $posthoc" }
    $arguments += @("--posthoc-metrics-csv", $posthoc)
}
& $Python @arguments
if ($LASTEXITCODE -ne 0) { throw "Terminal-Personal training/replay failed; resume using the same parameters and -Resume." }
