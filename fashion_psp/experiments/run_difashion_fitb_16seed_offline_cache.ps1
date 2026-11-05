param(
    [ValidateSet("bon", "psp", "all")][string]$Method = "all",
    [ValidateSet("0", "1", "2", "all")][string]$Group = "all",
    [ValidateRange(1, 2207)][int]$MaxSamples = 2207,
    [ValidateRange(1, 16)][int]$BonCandidates = 16,
    [ValidateRange(1, 16)][int]$PspCandidates = 16,
    [ValidateRange(1, 32)][int]$ConcurrentTasks = 1,
    [ValidateRange(0, 2206)][int]$ResumeFrom = 0,
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [string]$PythonExe = "D:\anaconda\envs\difashion\python.exe",
    [string]$DataRoot = "D:\DiFashion",
    [string]$CheckpointRoot = "D:\DiFashion",
    [string]$OutputRoot = "",
    [string]$Device = "cuda:0",
    [switch]$PrepareOnly
)
$ErrorActionPreference = "Stop"
if ($ResumeFrom -ge $MaxSamples) { throw "ResumeFrom must be smaller than MaxSamples." }
if (-not (Test-Path -LiteralPath $PythonExe)) { throw "Missing Python: $PythonExe" }
if (-not $OutputRoot) { $OutputRoot = Join-Path $RepoRoot "DiFashion-main\offline_fitb_strict50" }
$methods = if ($Method -eq "all") { @("bon", "psp") } else { @($Method) }
$groups = if ($Group -eq "all") { @(0, 1, 2) } else { @([int]$Group) }
$env:HF_HUB_OFFLINE = "1"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
Push-Location -LiteralPath $RepoRoot
try {
    foreach ($methodName in $methods) {
        $baseSeed = if ($methodName -eq "bon") { 42 } else { 2026 }
        $count = if ($methodName -eq "bon") { $BonCandidates } else { $PspCandidates }
        foreach ($groupId in $groups) {
            $groupSeed = $baseSeed + $groupId * 1000000
            $groupRoot = Join-Path (Join-Path $OutputRoot $methodName) "group_${groupId}_seed${groupSeed}_pool${count}"
            $arguments = @(
                "-m", "fashion_psp.experiments.run_difashion_fitb_16seed_offline_cache",
                "--method", $methodName, "--group-id", "$groupId", "--seed-base", "$baseSeed",
                "--num-particles", "$count", "--output-root", $groupRoot,
                "--data-root", $DataRoot, "--checkpoint-root", $CheckpointRoot,
                "--image-root", (Join-Path $DataRoot "ifashion\images"),
                "--base-model", (Join-Path $RepoRoot "DiFashion-main\pretrained_models\stable-diffusion-2-base"),
                "--max-tasks", "$MaxSamples", "--resume-from", "$ResumeFrom",
                "--device", $Device, "--concurrent-tasks", "$ConcurrentTasks"
            )
            if ($PrepareOnly) { $arguments += "--preflight-only" }
            Write-Host "[$methodName] group=$groupId base=$baseSeed group_seed=$groupSeed particles=$count tasks=$MaxSamples concurrent=$ConcurrentTasks"
            & $PythonExe @arguments
            if ($LASTEXITCODE -ne 0) { throw "DiFashion offline cache failed: $methodName group$groupId" }
        }
    }
} finally { Pop-Location }
