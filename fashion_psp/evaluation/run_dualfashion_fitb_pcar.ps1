param(
    [string]$RepoRoot = "E:\difashion-PSP",
    [string]$DataRoot = "C:\tmp\ifashion",
    [string]$Device = "cuda:0",
    [int]$MaxSamples = 2207,
    [int]$BatchSize = 4,
    [switch]$SkipSelection,
    [switch]$SkipPosthocMetrics
)

$ErrorActionPreference = "Stop"
$env:HF_HUB_OFFLINE = "1"
$dualRoot = Join-Path $RepoRoot "DualFashion-main"
$python = "D:\anaconda\envs\dualfashion\python.exe"
$pcar = Join-Path $RepoRoot "fashion_psp\artifacts\pcar\ifashion_train_v2\best.pt"
$outputRoot = Join-Path $dualRoot "output\ifashion_fitb_pcar"
$processed = Join-Path $dualRoot "processed_info"
$imageRoot = Join-Path $DataRoot "semantic_category"
$cache = Join-Path $dualRoot "output\ifashion_fitb_comp\psp_offline_cache"
$bonInput = Join-Path $dualRoot "output\ifashion_fitb_comp\bon\bon_group0_seed42"
$bonOutput = Join-Path $outputRoot "bon\bon_group0_seed42_scores"
$pspOutput = Join-Path $outputRoot "psp\psp_group0_pcar"

if (!(Test-Path -LiteralPath $python)) { throw "Python not found: $python" }
if (!(Test-Path -LiteralPath $pcar)) { throw "PCAR checkpoint not found: $pcar" }

Set-Location -LiteralPath $dualRoot

if (!$SkipSelection) {
    & $python .\score_bon_compatibility.py `
        --input-dir $bonInput `
        --data-root $DataRoot `
        --output-dir $bonOutput `
        --checkpoint 0057501 `
        --group-size 4 `
        --batch-size $BatchSize `
        --device $Device `
        --reward pcar `
        --pcar-checkpoint $pcar
    if ($LASTEXITCODE -ne 0) { throw "PCAR BoN selection failed." }

    & $python .\select_offline_psp_compatibility.py `
        --cache-root $cache `
        --group 0 `
        --base-seed 2026 `
        --data-root $DataRoot `
        --processed-info-root $processed `
        --image-root $imageRoot `
        --output-dir $pspOutput `
        --checkpoint 0057501 `
        --max-samples $MaxSamples `
        --batch-size $BatchSize `
        --device $Device `
        --reward pcar `
        --pcar-checkpoint $pcar `
        --resume-from auto
    if ($LASTEXITCODE -ne 0) { throw "PCAR PSP selection failed." }
}

$common = @(
    "--dataset", "ifashion", "--mode", "test", "--task", "FITB",
    "--data_path", $DataRoot,
    "--pretrained_evaluator_ckpt", (Join-Path $dualRoot "eval\compatibility_evaluator\ifashion-ckpt\ifashion_evaluator.pth"),
    "--cid_to_label_path", (Join-Path $dualRoot "eval\finetuned-inception\cid_to_label.npy"),
    "--inception_model_path", (Join-Path $dualRoot "eval\finetuned-inception\Inception-finetune-epoch300.zip"),
    "--batch_size", "1", "--num_workers", "0", "--device", $Device, "--lpips_net", "vgg", "--API_KEY="
)
$versions = @(
    "bon\bon_group0_seed42_scores\selected_for_official_eval\standard\group_1",
    "bon\bon_group0_seed42_scores\selected_for_official_eval\bon\group_1",
    "psp\psp_group0_pcar"
)
foreach ($version in $versions) {
    & $python .\evaluate_PFITB.py @common --output_dir $outputRoot --eval_version $version
    if ($LASTEXITCODE -ne 0) { throw "Official FITB evaluation failed: $version" }
}

if (!$SkipPosthocMetrics) {
    $gtSimRoot = Join-Path $dualRoot "output\ifashion_fitb_best_gt_similarity_with_pcar"
    $gaaRoot = Join-Path $dualRoot "output\ifashion_fitb_gt_attribute_alignment_with_pcar"
    & $python .\evaluate_best_results_gt_similarity.py --repo $dualRoot --data-root $DataRoot --output $gtSimRoot --device $Device --batch-size 32
    if ($LASTEXITCODE -ne 0) { throw "GT-Sim evaluation failed." }
    & $python .\evaluate_gt_attribute_alignment.py --repo $dualRoot --data-root $DataRoot --output $gaaRoot --device $Device --batch-size 24
    if ($LASTEXITCODE -ne 0) { throw "GAA evaluation failed." }
    & $python (Join-Path $RepoRoot "fashion_psp\evaluation\summarize_dualfashion_fitb_pcar.py") `
        --output-root $outputRoot `
        --gt-sim-summary (Join-Path $gtSimRoot "summary.csv") `
        --gaa-summary (Join-Path $gaaRoot "gaa_summary.csv") `
        --report-dir (Join-Path $outputRoot "all_metrics")
    if ($LASTEXITCODE -ne 0) { throw "PCAR metric summary failed." }
}

Write-Host "PCAR FITB evaluation complete: $outputRoot"
