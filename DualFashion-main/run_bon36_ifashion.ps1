param(
    [int]$MaxSamples = 10,
    [int]$NumSamples = 36,
    [int]$SampleBatchSize = 1,
    [int]$Seed = 2026,
    [int]$SeedStride = 1000,
    [int]$NumInferenceSteps = 28,
    [string]$OutputName = "pfitb_bon36_10tasks_seed2026"
)

$ErrorActionPreference = "Stop"

$python = "D:\anaconda\envs\dualfashion\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$outputRoot = Join-Path $repo ("output\" + $OutputName)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}
if ($NumSamples -ne 36) {
    throw "This wrapper is for the 36-candidate BoN experiment; use -NumSamples 36."
}
if ($MaxSamples -ne 10) {
    throw "This wrapper is configured for 10 tasks; use -MaxSamples 10."
}
if ($SampleBatchSize -lt 1 -or $SampleBatchSize -gt $NumSamples) {
    throw "SampleBatchSize must be between 1 and NumSamples."
}
if ($SeedStride -lt $NumSamples) {
    throw "SeedStride must be at least NumSamples."
}

Set-Location -LiteralPath $repo
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
& $python .\Inference_ifashion_PFITB.py `
    --models_root (Join-Path $repo "checkpoint") `
    --output_root $outputRoot `
    --image_root "C:\tmp\ifashion\semantic_category" `
    --data_root "C:\tmp\ifashion" `
    --fitb_file "processed\fitb_test.npy" `
    --processed_info_root (Join-Path $repo "processed_info") `
    --num_samples $NumSamples `
    --sample_batch_size $SampleBatchSize `
    --seed_stride $SeedStride `
    --max_samples $MaxSamples `
    --num_inference_steps $NumInferenceSteps `
    --text-guidance-scale 8 `
    --context-guidance-scale 7 `
    --preference-guidance-scale 8 `
    --seed $Seed `
    --deterministic_context `
    --skip_captions

if ($LASTEXITCODE -ne 0) {
    throw "PFITB inference failed with exit code $LASTEXITCODE."
}

$payload = Join-Path $outputRoot "0057501\gen.npy"
if (-not (Test-Path -LiteralPath $payload)) {
    throw "Inference finished without the expected payload: $payload"
}

Write-Host "BoN@36 generation completed: $outputRoot"
Write-Host "Candidates per task: $NumSamples; tasks: $MaxSamples; sample batch size: $SampleBatchSize"
