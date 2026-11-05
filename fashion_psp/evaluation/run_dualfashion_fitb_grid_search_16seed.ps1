param(
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\dualfashion\python.exe",
    [string]$MetricsCsv = "",
    [string]$OutputRoot = "",
    [int]$Dpi = 300,
    [ValidateSet("at-most", "exact")][string]$BudgetMode = "at-most",
    [switch]$PlotOnly
)
$ErrorActionPreference = "Stop"
$analysisRoot = Join-Path $RepoRoot "DualFashion-main\output\ifashion_fitb_timestep200_16particle_analysis"
if (-not $MetricsCsv) { $MetricsCsv = Join-Path $analysisRoot "candidate_metrics_personal_all_steps.csv" }
if (-not $OutputRoot) {
    $budgetLabel = if ($BudgetMode -eq "at-most") { "atmost" } else { "exact" }
    $OutputRoot = Join-Path $analysisRoot "grid_search\prefix_kset_${budgetLabel}112_first7"
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) { throw "Python not found: $PythonExe" }
$runArguments = @(
    (Join-Path $PSScriptRoot "plot_dualfashion_fitb_grid_search_16seed.py"),
    "--metrics-csv", $MetricsCsv, "--output-root", $OutputRoot,
    "--max-cutoffs", "4", "--dpi", "$Dpi", "--budget-mode", $BudgetMode
)
if ($PlotOnly) { $runArguments += "--plot-only" }
Write-Host "Retrospective 200-task search; prefix K=1,2,4,6,8,12,16. Not independent validation."
& $PythonExe @runArguments
if ($LASTEXITCODE -ne 0) { throw "16-seed grid search/plot failed." }
