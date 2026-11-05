param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$BatchSize = 4,
    [switch]$SkipSelection
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"

$python = "D:\anaconda\envs\dualfashion\python.exe"
$dual = Join-Path $RepoRoot "DualFashion-main"
$root = Join-Path $dual "output\ifashion_fitb_comp"
$history = Join-Path $DataRoot "processed\test_history_clipembs.npy"
$clip = "C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors"
$bonInput = Join-Path $root "bon\bon_group0_seed42"
$pspCache = Join-Path $root "psp_offline_cache"
$bonCompatibility = Join-Path $root "bon\bon_group0_seed42_scores"
$bonPersonal = Join-Path $root "bon\bon_group0_seed42_personal_scores"
$bonBottleneck = Join-Path $root "bon\bon_group0_seed42_bottleneck_scores"
$pspCompatibility = Join-Path $root "psp\psp_group0_compatibility"
# These runs already declare ifashion_fitb_comp/psp_offline_cache as their
# source cache and use the same group-0 seed pool/schedule.  Reuse the frozen
# payloads rather than replaying the 28-step cache a second time.
$pspPersonal = Join-Path $dual "output\ifashion_fitb_personal\psp\psp_group0_personal"
$pspBottleneck = Join-Path $dual "output\ifashion_fitb_bottleneck\psp\psp_group0_bottleneck"
$evaluator = Join-Path $RepoRoot "fashion_psp\evaluation\evaluate_dualfashion_fitb_native_metrics.py"
$output = Join-Path $root "native_metrics_reward_cross_validation"

function Test-Payload([string]$Path) {
    return (Test-Path -LiteralPath (Join-Path $Path "0057501\gen.npy")) -and (Test-Path -LiteralPath (Join-Path $Path "0057501\grd.npy"))
}

if (-not (Test-Path -LiteralPath $python)) { throw "Missing Python: $python" }
if (-not (Test-Path -LiteralPath $evaluator)) { throw "Missing evaluator: $evaluator" }

Set-Location -LiteralPath $dual

if (-not $SkipSelection) {
    foreach ($entry in @(
        @{ Reward = "personal"; Output = $bonPersonal },
        @{ Reward = "bottleneck"; Output = $bonBottleneck }
    )) {
        $payload = Join-Path $entry.Output "selected_for_official_eval\bon\group_1\0057501\gen.npy"
        if (-not (Test-Path -LiteralPath $payload)) {
            Write-Host "[BoN $($entry.Reward)] selecting from existing terminal candidates..."
            & $python .\score_bon_compatibility.py `
                --input-dir $bonInput --data-root $DataRoot --output-dir $entry.Output `
                --checkpoint 0057501 --group-size 4 --batch-size $BatchSize --device $Device `
                --reward $entry.Reward --history-path $history
            if ($LASTEXITCODE -ne 0) { throw "BoN $($entry.Reward) selection failed." }
        }
    }
    foreach ($entry in @(
        @{ Reward = "personal"; Output = $pspPersonal },
        @{ Reward = "bottleneck"; Output = $pspBottleneck }
    )) {
        if (-not (Test-Payload $entry.Output)) {
            Write-Host "[PSP $($entry.Reward)] replaying only cached step 7/14/28 candidates..."
            & $python .\select_offline_psp_compatibility.py `
                --cache-root $pspCache --group 0 --base-seed 2026 --data-root $DataRoot `
                --processed-info-root (Join-Path $dual "processed_info") `
                --image-root (Join-Path $DataRoot "semantic_category") `
                --output-dir $entry.Output --checkpoint 0057501 --batch-size $BatchSize --device $Device `
                --reward $entry.Reward --history-path $history --clip-weights $clip --resume-from auto
            if ($LASTEXITCODE -ne 0) { throw "PSP $($entry.Reward) selection failed." }
        }
    }
}

$standard = Join-Path $bonCompatibility "selected_for_official_eval\standard\group_1\0057501"
$bonCompatSelected = Join-Path $bonCompatibility "selected_for_official_eval\bon\group_1\0057501"
$bonPersonalSelected = Join-Path $bonPersonal "selected_for_official_eval\bon\group_1\0057501"
$bonBottleneckSelected = Join-Path $bonBottleneck "selected_for_official_eval\bon\group_1\0057501"
$inputs = @(
    "standard=$standard",
    "compatibility/bon=$bonCompatSelected",
    "compatibility/psp=$(Join-Path $pspCompatibility '0057501')",
    "personal/bon=$bonPersonalSelected",
    "personal/psp=$(Join-Path $pspPersonal '0057501')",
    "bottleneck/bon=$bonBottleneckSelected",
    "bottleneck/psp=$(Join-Path $pspBottleneck '0057501')"
)

foreach ($entry in $inputs) {
    $payload = ($entry -split "=", 2)[1]
    if (-not (Test-Path -LiteralPath (Join-Path $payload "gen.npy"))) { throw "Missing frozen selection payload: $payload" }
}

$selectionArgs = foreach ($entry in $inputs) {
    "--selection"
    $entry
}

Write-Host "[Native metrics] frozen selected images only; GT is post-selection for CIS/LPIPS."
& $python $evaluator @selectionArgs --output-root $output --dualfashion-root $dual --data-root $DataRoot --device $Device --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw "Native FITB metric evaluation failed." }
