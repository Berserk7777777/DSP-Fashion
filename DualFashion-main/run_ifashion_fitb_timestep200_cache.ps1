param(
    [ValidateRange(0, 2)]
    [int]$Group = 0,
    [string]$ResumeFrom = "auto",
    [ValidateRange(1, 2207)]
    [int]$MaxSamples = 200,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"

$python = "D:\anaconda\envs\dualfashion\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$dataRoot = "C:\tmp\ifashion"
$imageRoot = "C:\tmp\ifashion\semantic_category"
$processedInfoRoot = Join-Path $repo "processed_info"
$modelRoot = Join-Path $repo "checkpoint"
$outputRoot = Join-Path $repo "output\ifashion_fitb_timestep200_cache"
$pspSeeds = @(2026, 1002026, 2002026)
$allSteps = [string]::Join(",", @(1..28))

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}

Set-Location -LiteralPath $repo
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:HF_HUB_OFFLINE = "1"

$seed = $pspSeeds[$Group]
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

& $python .\run_ifashion_offline_cache.py `
    --models-root $modelRoot `
    --model-name 0057501 `
    --data-root $dataRoot `
    --fitb-file processed\fitb_test.npy `
    --processed-info-root $processedInfoRoot `
    --image-root $imageRoot `
    --output-root $outputRoot `
    --group $Group `
    --max-samples $MaxSamples `
    --num-particles 8 `
    --num-inference-steps 28 `
    --offline-cache-steps $allSteps `
    --seed $seed `
    --seed-stride 10000 `
    --text-guidance-scale 8 `
    --context-guidance-scale 7 `
    --preference-guidance-scale 8 `
    --device $Device `
    --cpu-offload `
    --deterministic-context `
    --resume-from $ResumeFrom

if ($LASTEXITCODE -ne 0) {
    throw "All-timestep offline cache inference failed for group $Group with exit code $LASTEXITCODE."
}

Write-Host "Completed all-timestep FITB cache: group=$Group tasks=$MaxSamples output=$outputRoot"
