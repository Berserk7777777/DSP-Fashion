param(
    [ValidateSet("bon", "psp", "all")]
    [string]$Method = "all",
    [ValidateRange(0, 2)]
    [int]$Group = 0,
    [string]$ResumeFrom = "auto",
    [int]$MaxSamples = 2207,
    [switch]$NoAuditAllCandidates
)

$ErrorActionPreference = "Stop"

$python = "D:\anaconda\envs\dualfashion\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$dataRoot = "C:\tmp\ifashion"
$imageRoot = "C:\tmp\ifashion\semantic_category"
$processedInfoRoot = Join-Path $repo "processed_info"
$outputRoot = Join-Path $repo "output\ifashion_fitb_comp"
$modelRoot = Join-Path $repo "checkpoint"
$modelName = "0057501"
$seedStride = 10000
$numInferenceSteps = 28

$bonSeeds = @(42, 1000042, 2000042)
$pspSeeds = @(2026, 1002026, 2002026)

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}
if ($MaxSamples -lt 1 -or $MaxSamples -gt 2207) {
    throw "MaxSamples must be between 1 and 2207."
}

$useResume = -not [string]::IsNullOrWhiteSpace($ResumeFrom) -and $ResumeFrom.ToLowerInvariant() -ne "none"
$resumeStart = 0
if ($useResume -and $ResumeFrom.ToLowerInvariant() -ne "auto") {
    if ($ResumeFrom -notmatch '^\d+$') {
        throw "ResumeFrom must be auto, none, or a non-negative task index."
    }
    $resumeStart = [int]$ResumeFrom
}
if ($resumeStart -ge $MaxSamples) {
    throw "ResumeFrom ($resumeStart) must be smaller than MaxSamples ($MaxSamples)."
}

Set-Location -LiteralPath $repo
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
$env:HF_HUB_OFFLINE = "1"

function Test-PspTaskComplete {
    param(
        [Parameter(Mandatory = $true)][string]$TaskDir,
        [Parameter(Mandatory = $true)][bool]$AuditAllCandidates
    )

    $metadataPath = Join-Path $TaskDir "metadata.json"
    if (-not (Test-Path -LiteralPath $metadataPath)) {
        return $false
    }
    try {
        $metadata = Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }
    $finalRows = @($metadata.final)
    if ($finalRows.Count -ne 1) {
        return $false
    }
    if (-not (Test-Path -LiteralPath ([string]$finalRows[0].image_path))) {
        return $false
    }
    if ($AuditAllCandidates) {
        $auditRows = @($metadata.audit_candidates)
        if (-not [bool]$metadata.audit_all_candidates -or $auditRows.Count -ne 8) {
            return $false
        }
    }
    return $true
}

function Invoke-BonGroup {
    param([int]$GroupIndex)

    $seed = $bonSeeds[$GroupIndex]
    $runRoot = Join-Path $outputRoot "bon\bon_group${GroupIndex}_seed${seed}"
    New-Item -ItemType Directory -Force -Path $runRoot | Out-Null
    $pythonArgs = @(
        ".\Inference_ifashion_PFITB.py",
        "--models_root", $modelRoot,
        "--output_root", $runRoot,
        "--image_root", $imageRoot,
        "--data_root", $dataRoot,
        "--fitb_file", "processed\fitb_test.npy",
        "--processed_info_root", $processedInfoRoot,
        "--num_samples", "4",
        "--sample_batch_size", "1",
        "--seed_stride", $seedStride.ToString(),
        "--max_samples", $MaxSamples.ToString(),
        "--num_inference_steps", $numInferenceSteps.ToString(),
        "--text-guidance-scale", "8",
        "--context-guidance-scale", "7",
        "--preference-guidance-scale", "8",
        "--seed", $seed.ToString(),
        "--cpu_offload",
        "--deterministic_context",
        "--skip_captions"
    )
    if ($useResume) {
        $pythonArgs += @("--resume-from", $ResumeFrom)
    }

    Write-Host "[BoN] group=$GroupIndex seed=$seed output=$runRoot resume=$ResumeFrom"
    & $python @pythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "BoN inference failed for group $GroupIndex with exit code $LASTEXITCODE."
    }
}

function Invoke-PspGroup {
    param([int]$GroupIndex)

    $seed = $pspSeeds[$GroupIndex]
    $groupRoot = Join-Path $outputRoot "psp"
    New-Item -ItemType Directory -Force -Path $groupRoot | Out-Null
    $auditAllCandidates = -not $NoAuditAllCandidates.IsPresent

    for ($taskIndex = 0; $taskIndex -lt $MaxSamples; $taskIndex++) {
        if ($useResume -and $taskIndex -lt $resumeStart) {
            continue
        }

        $taskName = "psp_group${GroupIndex}_task${taskIndex}_seed${seed}"
        $taskDir = Join-Path $groupRoot $taskName
        if ($useResume -and (Test-PspTaskComplete -TaskDir $taskDir -AuditAllCandidates $auditAllCandidates)) {
            Write-Host "[PSP] skip completed task=$taskIndex group=$GroupIndex"
            continue
        }

        $pythonArgs = @(
            ".\run_psp_ifashion_smoke.py",
            "--models-root", $modelRoot,
            "--model-name", $modelName,
            "--data-root", $dataRoot,
            "--fitb-file", "processed\fitb_test.npy",
            "--processed-info-root", $processedInfoRoot,
            "--image-root", $imageRoot,
            "--output-dir", $taskDir,
            "--task-index", $taskIndex.ToString(),
            "--num-particles", "8",
            "--num-inference-steps", $numInferenceSteps.ToString(),
            "--prune-schedule", "7:4,14:2,28:1",
            "--seed", $seed.ToString(),
            "--seed-stride", $seedStride.ToString(),
            "--cpu-offload",
            "--deterministic-context"
        )
        if ($auditAllCandidates) {
            # Deliberately omit --save-intermediate-predictions.  The final
            # experiment keeps final/all-candidate audit images but no latent
            # projections or intermediate_prediction_artifacts.csv.
            $pythonArgs += "--audit-all-candidates"
        }

        Write-Host "[PSP] group=$GroupIndex task=$taskIndex/$MaxSamples seed=$seed"
        & $python @pythonArgs
        if ($LASTEXITCODE -ne 0) {
            throw "PSP inference failed for group $GroupIndex task $taskIndex with exit code $LASTEXITCODE."
        }
    }
}

if ($Method -in @("bon", "all")) {
    Invoke-BonGroup -GroupIndex $Group
}
if ($Method -in @("psp", "all")) {
    Invoke-PspGroup -GroupIndex $Group
}

Write-Host "Completed method=$Method group=$Group max_samples=$MaxSamples output=$outputRoot"
