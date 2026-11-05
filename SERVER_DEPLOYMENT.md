# Server deployment

## Clone and environments

```bash
git clone https://github.com/Berserk7777777/DSP-Fashion.git
cd DSP-Fashion
```

The upload helper creates a private repository. Authenticate your server's
Git client before cloning; do not embed a token in its URL.

Use a separate environment for each backbone. Current DiFashion and
DualFashion workflows were verified locally on Python 3.11 with PyTorch
2.8.0+cu128 and torchvision 0.23.0+cu128. Choose a CUDA wheel suitable for the
server's driver, then install the recorded core packages:

```bash
conda create -n difashion python=3.11 -y
conda activate difashion
# Example wheel for a driver that supports CUDA 12.8:
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/difashion-core.txt
python -m pip install -e .

conda create -n dualfashion python=3.11 -y
conda activate dualfashion
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environments/dualfashion-core.txt
python -m pip install -e .
```

These core files record directly used working versions. They are not full
lockfiles or a claim of Linux GPU validation. Original upstream requirements
reflect older environments. FashionDPO retains its separate
`FashionDPO-main/fashiondpo_environment.yml`; its shared DSP integration has
not been verified. Do not install all backbone environments together.

## External assets

Transfer datasets, model weights, and generated caches separately. They are
excluded from Git. Preserve full native model directories, including configs.

DiFashion generation needs:

```text
/data/DiFashion/
├── checkpoint-15000/
│   ├── unet_ema/
│   └── fashion_encoder_ema/
└── ifashion/
    ├── processed/new_fitb_test.npy
    ├── processed/test_hist_latents.npy
    ├── new_id_cate_dict.npy
    ├── new_all_item_image_paths.npy
    └── images/                  # including empty_image.png
/models/stable-diffusion-2-base/  # complete pipeline
```

DualFashion needs its full `checkpoint/0057501/`, `pretrained_models/`,
`processed_info/`, and native iFashion dataset tree. Its collection launcher
lists the required assets and arguments:
`fashion_psp/experiments/run_dualfashion_fitb_16seed_offline_cache.ps1`.
On Linux invoke `DualFashion-main/run_ifashion_offline_cache.py` with the
equivalent explicit paths; keep the launcher's `offline_cache_protocol.json`
alongside each group when moving existing caches for evaluation.

Additional evaluation assets:

- `cnn_features_clip.npy` and `processed/test_history_clipembs.npy` in each
  dataset root, with the corresponding GT and image lookup arrays.
- Each backbone's native `compatibility_evaluator/ifashion-ckpt/ifashion_evaluator.pth`:
  under `DiFashion-main/Evaluation/` and `DualFashion-main/eval/` respectively.
- OpenCLIP ViT-H-14/laion2b-s32b-b79K safetensors, torchvision's cached VGG16
  backbone weights, and LPIPS's VGG calibration weights. Provision these
  before using offline Hugging Face mode.

## DiFashion collection example

Activate `difashion` and first run a preflight:

```bash
python -m fashion_psp.experiments.run_difashion_fitb_16seed_offline_cache \
  --method psp --group-id 0 --seed-base 2026 --num-particles 16 \
  --max-tasks 2207 --concurrent-tasks 2 \
  --data-root /data/DiFashion --checkpoint-root /data/DiFashion \
  --base-model /models/stable-diffusion-2-base \
  --output-root /outputs/offline_fitb_strict50/psp/group_0_seed2026_pool16 \
  --device cuda:0 --resume-from 0 --preflight-only
```

Remove `--preflight-only` for collection. For BoN use `--method bon
--seed-base 42` and a separate output directory; it saves only the terminal
image. Repeat for `--group-id 1` and `2`, using separate output directories.
`--seed-base` is the method base before the group offset; do not add the
offset twice. Each concurrent task worker loads a separate model instance;
reduce concurrency if VRAM is insufficient. Within each task seeds are serial.

## Offline metrics

Use the appropriate backbone environment and the native data roots:

```bash
python -m fashion_psp.evaluation.evaluate_fitb_offline_pools \
  --model difashion --cache-root /outputs/offline_fitb_strict50 \
  --data-root /data/DiFashion/ifashion \
  --personal-clip-weights /models/open_clip_model.safetensors \
  --groups 0,1,2 --max-tasks 2207 --batch-size 4 --device cuda:0

python -m fashion_psp.evaluation.evaluate_fitb_offline_pools \
  --model dualfashion --cache-root /outputs/ifashion_fitb_16seed_offline \
  --data-root /data/DualFashion/ifashion \
  --personal-clip-weights /models/open_clip_model.safetensors \
  --groups 0,1,2 --max-tasks 2207 --batch-size 4 --device cuda:0
```

Generated trajectories and metric databases are not part of a fresh clone.
Transfer existing cache group directories, including metadata/protocol files,
before these commands. DualFashion's Windows image paths are resolved against
their relocated task directory. Evaluation never generates diffusion images.
The evaluation guide documents partial caches, resume, and all Excel tables:
[OFFLINE_FITB_POOL_METRICS.md](fashion_psp/evaluation/OFFLINE_FITB_POOL_METRICS.md).
