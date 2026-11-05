param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 4,
    [string]$CondaEnvironment = "dualfashion",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$dualFashionRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $RepoRoot "fashion_psp\evaluation\score_dualfashion_fitb_terminal_gt.py"
$analyzer = Join-Path $RepoRoot "fashion_psp\evaluation\analyze_dualfashion_fitb_regret.py"
$cacheRoot = Join-Path $dualFashionRoot "output\ifashion_fitb_comp\psp_offline_cache"
$outputRoot = Join-Path $dualFashionRoot "output\ifashion_fitb_terminal_regret"
$conda = Get-Command conda -ErrorAction Stop

$scoreArgs = @(
    $runner,
    "--cache-root", $cacheRoot,
    "--data-root", $DataRoot,
    "--output-root", $outputRoot,
    "--batch-size", "$BatchSize",
    "--device", $Device,
    "--max-samples", "2207"
)
if ($Resume) { $scoreArgs += "--resume" }
& $conda.Source run --no-capture-output --name $CondaEnvironment python @scoreArgs
if ($LASTEXITCODE -ne 0) { throw "Terminal GT diagnostic scoring failed." }

& $conda.Source run --no-capture-output --name $CondaEnvironment python $analyzer `
    "--dualfashion-root" $dualFashionRoot `
    "--terminal-gt-metrics" (Join-Path $outputRoot "terminal_gt_candidate_metrics.csv") `
    "--output-root" $outputRoot
if ($LASTEXITCODE -ne 0) { throw "Terminal regret analysis failed." }
