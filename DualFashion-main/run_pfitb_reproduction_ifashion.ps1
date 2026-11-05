param(
    [int]$MaxSamples = 10,
    [int]$Seed = 2026,
    [int]$NumInferenceSteps = 28,
    [int]$CaptionSteps = 64,
    [int]$SeedStride = 1000,
    [string]$OutputName = "pfitb_reproduction_text_10_seed2026"
)

$ErrorActionPreference = "Stop"
$python = "D:\anaconda\envs\dualfashion\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$outputRoot = Join-Path $repo ("output\" + $OutputName)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}

Set-Location -LiteralPath $repo
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$arguments = @(
    ".\Inference_ifashion_PFITB.py",
    "--models_root", (Join-Path $repo "checkpoint"),
    "--output_root", $outputRoot,
    "--data_root", "C:\tmp\ifashion",
    "--fitb_file", "processed\fitb_test.npy",
    "--processed_info_root", (Join-Path $repo "processed_info"),
    "--image_root", "C:\tmp\ifashion\semantic_category",
    "--max_samples", $MaxSamples,
    "--num_samples", 1,
    "--sample_batch_size", 1,
    "--seed_stride", $SeedStride,
    "--num_caption_samples", 1,
    "--num_inference_steps", $NumInferenceSteps,
    "--caption_steps", $CaptionSteps,
    "--seed", $Seed,
    "--cpu_offload",
    "--deterministic_context"
)

& $python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "PFITB reproduction inference failed with exit code $LASTEXITCODE."
}

$payload = Join-Path $outputRoot "0057501\gen.npy"
if (-not (Test-Path -LiteralPath $payload)) {
    throw "Inference finished without the expected payload: $payload"
}

Write-Host "PFITB image+text reproduction completed: $outputRoot"
Write-Host "num_samples=1, num_caption_samples=1, captions enabled"
