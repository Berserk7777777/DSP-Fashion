param(
    [string]$EvalVersion = "pfitb_reproduction_text_10_seed2026",
    [string]$OutputDir = "E:\difashion-PSP\DualFashion-main\output",
    [int]$BatchSize = 1,
    [string]$Device = "cuda:0"
)

$ErrorActionPreference = "Stop"
$python = "D:\anaconda\envs\svdf\python.exe"
$repo = "E:\difashion-PSP\DualFashion-main"
$evaluator = Join-Path $repo "eval\compatibility_evaluator\ifashion-ckpt\ifashion_evaluator.pth"
$cidToLabel = Join-Path $repo "eval\finetuned-inception\cid_to_label.npy"
$inception = Join-Path $repo "eval\finetuned-inception\Inception-finetune-epoch300.zip"

Set-Location -LiteralPath $repo
& $python .\evaluate_PFITB.py `
    --dataset ifashion `
    --mode test `
    --task FITB `
    --data_path C:\tmp\ifashion `
    --pretrained_evaluator_ckpt $evaluator `
    --cid_to_label_path $cidToLabel `
    --inception_model_path $inception `
    --output_dir $OutputDir `
    --eval_version $EvalVersion `
    --batch_size $BatchSize `
    --num_workers 0 `
    --device $Device `
    --lpips_net vgg `
    --API_KEY ""

if ($LASTEXITCODE -ne 0) {
    throw "PFITB evaluation failed with exit code $LASTEXITCODE."
}
