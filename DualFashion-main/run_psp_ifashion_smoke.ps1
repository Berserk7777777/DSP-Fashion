param(
    [int]$TaskIndex = 0,
    [int]$NumParticles = 8,
    [int]$NumInferenceSteps = 28,
    [string]$PruneSchedule = "7:4,14:2,28:1",
    [int]$Seed = 2026,
    [int]$SeedStride = 10000,
    [string]$OutputName = "psp_task0_n8_7x4_14x2_28x1_seed2026"
)

$ErrorActionPreference = "Stop"
$python = "D:\anaconda\envs\difashion\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$outputDir = Join-Path $repo ("output\" + $OutputName)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}
if ($NumParticles -lt 1) {
    throw "NumParticles must be positive."
}

Set-Location -LiteralPath $repo
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& $python .\run_psp_ifashion_smoke.py `
    --models-root (Join-Path $repo "checkpoint") `
    --model-name "0057501" `
    --data-root "C:\tmp\ifashion" `
    --fitb-file "processed\fitb_test.npy" `
    --processed-info-root (Join-Path $repo "processed_info") `
    --image-root "C:\tmp\ifashion\semantic_category" `
    --output-dir $outputDir `
    --task-index $TaskIndex `
    --num-particles $NumParticles `
    --num-inference-steps $NumInferenceSteps `
    --prune-schedule $PruneSchedule `
    --seed $Seed `
    --seed-stride $SeedStride `
    --audit-all-candidates `
    --save-intermediate-predictions `
    --cpu-offload `
    --deterministic-context

if ($LASTEXITCODE -ne 0) {
    throw "DualFashion PSP smoke failed with exit code $LASTEXITCODE."
}

$metadata = Join-Path $outputDir "metadata.json"
if (-not (Test-Path -LiteralPath $metadata)) {
    throw "PSP smoke finished without metadata: $metadata"
}

Write-Host "DualFashion PSP smoke completed: $outputDir"
Write-Host "task=$TaskIndex; particles=$NumParticles; schedule=$PruneSchedule; steps=$NumInferenceSteps"
