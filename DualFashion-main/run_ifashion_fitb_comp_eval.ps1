param(
    [string]$Groups = "0",
    [switch]$SkipScoring
)

$ErrorActionPreference = "Stop"

$python = "D:\anaconda\envs\dualfashion\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$dataRoot = "C:\tmp\ifashion"
$imageRoot = "C:\tmp\ifashion\semantic_category"
$processedInfoRoot = Join-Path $repo "processed_info"
$outputRoot = Join-Path $repo "output\ifashion_fitb_comp"
$evaluator = Join-Path $repo "eval\compatibility_evaluator\ifashion-ckpt\ifashion_evaluator.pth"
$cidToLabel = Join-Path $repo "eval\finetuned-inception\cid_to_label.npy"
$inception = Join-Path $repo "eval\finetuned-inception\Inception-finetune-epoch300.zip"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python"
}

$groupIndices = @($Groups -split "," | ForEach-Object {
    $value = $_.Trim()
    if ($value -notmatch '^[0-2]$') {
        throw "Groups must be a comma-separated list containing only 0, 1, and 2."
    }
    [int]$value
})
if ($groupIndices.Count -eq 0) {
    throw "At least one group is required."
}

$bonSeeds = @(42, 1000042, 2000042)
$pspSeeds = @(2026, 1002026, 2002026)

Set-Location -LiteralPath $repo
$env:HF_HUB_OFFLINE = "1"

$common = @(
    "--dataset", "ifashion",
    "--mode", "test",
    "--task", "FITB",
    "--data_path", $dataRoot,
    "--pretrained_evaluator_ckpt", $evaluator,
    "--cid_to_label_path", $cidToLabel,
    "--inception_model_path", $inception,
    "--batch_size", "1",
    "--num_workers", "0",
    "--device", "cuda:0",
    "--lpips_net", "vgg",
    "--API_KEY="
)

foreach ($group in $groupIndices) {
    $bonSeed = $bonSeeds[$group]
    $pspSeed = $pspSeeds[$group]
    $bonRunRoot = Join-Path $outputRoot "bon\bon_group${group}_seed${bonSeed}"
    $bonScoreRoot = Join-Path $outputRoot "bon\bon_group${group}_seed${bonSeed}_scores"
    $pspRoot = Join-Path $outputRoot "psp"

    if (-not $SkipScoring.IsPresent) {
        Write-Host "[Score BoN] group=$group seed=$bonSeed"
        & $python .\score_bon_compatibility.py `
            --input-dir $bonRunRoot `
            --data-root $dataRoot `
            --output-dir $bonScoreRoot `
            --checkpoint 0057501 `
            --group-size 4 `
            --batch-size 1 `
            --device cuda:0
        if ($LASTEXITCODE -ne 0) {
            throw "BoN scoring failed for group $group."
        }

        Write-Host "[Pack PSP] group=$group seed=$pspSeed"
        & $python .\pack_psp_for_official_eval.py `
            --output-root $pspRoot `
            --group $group `
            --base-seed $pspSeed `
            --output-dir (Join-Path $pspRoot "psp_group${group}_official_eval") `
            --data-root $dataRoot `
            --processed-info-root $processedInfoRoot `
            --image-root $imageRoot
        if ($LASTEXITCODE -ne 0) {
            throw "PSP packing failed for group $group."
        }
    }

    $versions = @(
        @{
            Label = "Standard"
            EvalVersion = "bon\bon_group${group}_seed${bonSeed}_scores\selected_for_official_eval\standard\group_1"
        },
        @{
            Label = "BoN"
            EvalVersion = "bon\bon_group${group}_seed${bonSeed}_scores\selected_for_official_eval\bon\group_1"
        },
        @{
            Label = "PSP"
            EvalVersion = "psp\psp_group${group}_official_eval"
        }
    )

    foreach ($entry in $versions) {
        Write-Host "[Official Eval] group=$group method=$($entry.Label)"
        & $python .\evaluate_PFITB.py `
            @common `
            --output_dir $outputRoot `
            --eval_version $entry.EvalVersion
        if ($LASTEXITCODE -ne 0) {
            throw "Official evaluation failed for group $group method $($entry.Label)."
        }
    }
}

Write-Host "Official evaluation completed for groups: $Groups"
