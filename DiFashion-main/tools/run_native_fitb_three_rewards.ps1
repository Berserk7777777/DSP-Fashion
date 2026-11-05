param(
    [int]$WaitForPid = 0,
    [int]$BatchSize = 16
)

$ErrorActionPreference = 'Stop'

$root = 'E:\difashion-PSP\DiFashion-main'
$python = 'D:\anaconda\envs\difashion\python.exe'
$script = Join-Path $root 'tools\evaluate_fitb_offline_native.py'
$cache = Join-Path $root 'fitb_ifashion_offline_data'
$scores = Join-Path $root 'offline_native_eval_50step_shared_scores_v1\candidate_scores'
$common = @(
    $script,
    '--cache-root', $cache,
    '--data-root', 'D:\DiFashion\ifashion',
    '--evaluation-root', (Join-Path $root 'Evaluation'),
    '--score-root', $scores,
    '--device', 'cuda:0',
    '--batch-size', $BatchSize,
    '--resume'
)

if ($WaitForPid -gt 0) {
    try {
        Wait-Process -Id $WaitForPid -ErrorAction Stop
    }
    catch {
        Write-Warning "Candidate-score process $WaitForPid is not running; each reward stage will resume or finish the shared score cache."
    }
}

foreach ($reward in @('compatibility', 'personal', 'bottleneck')) {
    $output = Join-Path $root ("offline_native_eval_50step_{0}_group0_v1" -f $reward)
    Write-Host "Starting native DiFashion FITB evaluation for reward=$reward"
    & $python @common '--output-root' $output '--reward' $reward
    if ($LASTEXITCODE -ne 0) {
        throw "Native FITB evaluation failed for reward=$reward with exit code $LASTEXITCODE."
    }
}
