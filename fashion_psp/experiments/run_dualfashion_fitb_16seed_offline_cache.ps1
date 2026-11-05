param(
    [ValidateSet("bon", "psp", "all")]
    [string]$Method = "all",
    [ValidateSet("0", "1", "2", "all")]
    [string]$Group = "all",
    [ValidateRange(1, 2207)]
    [int]$MaxSamples = 2207,
    [ValidateRange(1, 32)]
    [int]$ConcurrentTasks = 1,
    [string]$ResumeFrom = "auto",
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\dualfashion\python.exe",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$OutputRoot = "",
    [string]$Device = "cuda:0",
    [switch]$PrepareOnly
)

$ErrorActionPreference = "Stop"
$dualRoot = Join-Path $RepoRoot "DualFashion-main"
$runner = Join-Path $dualRoot "run_ifashion_offline_cache.py"
$modelRoot = Join-Path $dualRoot "checkpoint"
$processedRoot = Join-Path $dualRoot "processed_info"
if (-not $OutputRoot) {
    $OutputRoot = Join-Path $dualRoot "output\ifashion_fitb_16seed_offline"
}
foreach ($path in @($PythonExe, $runner, (Join-Path $modelRoot "0057501"),
                   (Join-Path $DataRoot "processed\fitb_test.npy"),
                   (Join-Path $DataRoot "test_grd_dict.npy"),
                   (Join-Path $DataRoot "id_cate_dict.npy"),
                   (Join-Path $DataRoot "semantic_category"),
                   (Join-Path $processedRoot "all_item_image_paths.npy"),
                   (Join-Path $processedRoot "stage2_ifashion_test\test_preference.npy"))) {
    if (-not (Test-Path -LiteralPath $path)) { throw "Missing required path: $path" }
}
if ($ResumeFrom -ne "auto" -and $ResumeFrom -ne "none") {
    if ($ResumeFrom -notmatch '^\d+$' -or [int]$ResumeFrom -ge $MaxSamples) {
        throw "ResumeFrom must be auto, none, or an index smaller than MaxSamples."
    }
}
$methods = if ($Method -eq "all") { @("bon", "psp") } else { @($Method) }
$groups = if ($Group -eq "all") { @(0, 1, 2) } else { @([int]$Group) }
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"

foreach ($methodName in $methods) {
    $baseSeed = if ($methodName -eq "bon") { 42 } else { 2026 }
    $cacheSteps = if ($methodName -eq "bon") { "28" } else { [string]::Join(",", @(1..28)) }
    $viewsPerTask = if ($methodName -eq "bon") { 16 } else { 448 }
    foreach ($groupId in $groups) {
        $groupSeed = $baseSeed + $groupId * 1000000
        $groupRoot = Join-Path (Join-Path $OutputRoot $methodName) "group_${groupId}_seed${groupSeed}"
        $runArguments = @(
            $runner, "--models-root", $modelRoot, "--model-name", "0057501",
            "--data-root", $DataRoot, "--fitb-file", "processed\fitb_test.npy",
            "--processed-info-root", $processedRoot,
            "--image-root", (Join-Path $DataRoot "semantic_category"),
            "--output-root", $groupRoot, "--group", "$groupId",
            "--max-samples", "$MaxSamples", "--concurrent-tasks", "$ConcurrentTasks", "--num-particles", "16",
            "--num-inference-steps", "28", "--offline-cache-steps", $cacheSteps,
            "--seed", "$groupSeed", "--seed-stride", "10000",
            "--text-guidance-scale", "8", "--context-guidance-scale", "7",
            "--preference-guidance-scale", "8", "--device", $Device,
            "--cpu-offload", "--deterministic-context", "--resume-from", $ResumeFrom
        )
        Write-Host "[$methodName] group=$groupId seed=$groupSeed tasks=$MaxSamples concurrent_tasks=$ConcurrentTasks candidates=16 saved_images_per_task=$viewsPerTask"
        Write-Host "Output: $groupRoot"
        if ($PrepareOnly) { continue }
        New-Item -ItemType Directory -Force -Path $groupRoot | Out-Null
        @{
            method = $methodName; group_id = $groupId; base_seed = $baseSeed;
            group_seed = $groupSeed; seed_stride = 10000; num_particles = 16;
            num_inference_steps = 28; cache_steps = $cacheSteps; max_samples = $MaxSamples;
            concurrent_tasks = $ConcurrentTasks;
            seed_mapping = "base_seed + group_id*1000000 + task_index*10000 + slot (0..15)";
            runner_sha256 = (Get-FileHash -LiteralPath $runner -Algorithm SHA256).Hash;
            python_arguments = $runArguments
        } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $groupRoot "offline_cache_protocol.json") -Encoding UTF8
        & $PythonExe @runArguments
        if ($LASTEXITCODE -ne 0) {
            throw "FITB 16-seed cache failed: method=$methodName group=$groupId; rerun with -ResumeFrom auto."
        }
    }
}
Write-Host "Finished requested FITB 16-seed cache jobs. PrepareOnly=$PrepareOnly"
