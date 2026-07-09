# Grimoire — Reproduction & Operations Guide

> Working notes for reinstalling, locating checkpoints/data, and running the
> experiments for the **Grimoire** thesis codebase
> (GitHub `DerEchteFeuerpfeil/thesis`, wandb project `grimoire-2`).
> Compiled 2026-07-07 by inspecting the code + this server's filesystem.
> The repo's `README.md` describes the *original* training machine; **its paths
> are stale**. This file records what is actually true on the current server.

---

## 0. TL;DR — the single most important fact

Every path in `README.md` and in the `configs/*.yaml` files points at
`/raid/marco.cipriano/...`. **That mount does not exist on this server.**
The results and data were migrated to a **third storage root** (not the repo
root, not the `sci-demelo` root) that only appears in shell history:

```
/raid/marco.cipriano            →  /sc/projects/sci-aisc/marco.cipriano
```

This is a **1:1 prefix swap** — the `results/svg/Grimoire/{VSQ,ART}/...` and
`data/SVG/Grimoire/...` sub-structure is byte-for-byte identical. So:

| README / config says | Real location on this server |
|---|---|
| `/raid/marco.cipriano/results/svg/Grimoire/VSQ/...` | `/sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/VSQ/...` |
| `/raid/marco.cipriano/results/svg/Grimoire/ART/...` | `/sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/ART/...` |
| `/raid/marco.cipriano/data/SVG/Grimoire/...` | `/sc/projects/sci-aisc/marco.cipriano/data/SVG/Grimoire/...` |

**Fastest fix to run the code unmodified** — symlink the old prefix:

```bash
sudo mkdir -p /raid/marco.cipriano   # if you have root; else edit configs instead
ln -s /sc/projects/sci-aisc/marco.cipriano/results /raid/marco.cipriano/results
ln -s /sc/projects/sci-aisc/marco.cipriano/data    /raid/marco.cipriano/data
```

If you cannot create `/raid`, rewrite the prefix in every config you use
(`logging_params.save_dir`, `data_params.data_path`/`csv_path`/`vq_token_npy_path`,
`stage1_params.checkpoint_path`, `exp_params.continue_checkpoint`).

**Verdict on the four key questions:**
1. **Environment** — *not currently installed*; recreate from `requirements.yaml`
   + rebuild `diffvg` from source (details in §1). ⚠️ several gotchas.
2. **Checkpoints** — ✅ **fully recoverable** for VSQ (stage 1) and ART (stage 2)
   on FIGR8, FONTS, MNIST (BW+color) under the `sci-aisc` results root (§2).
3. **Datasets** — ✅ on disk under the `sci-aisc` data root; ❌ **not published
   on HuggingFace** (only private collaborator backups exist) (§3).
4. **Launchers** — `run.py` = stage 1 (VSQ), `run_stage2.py` = stage 2 (ART);
   copy-paste commands in §4.

---

## 1. Environment & installation

### 1.1 State on this server
- **No `SVG` conda env exists.** `conda env list` shows only
  `base, elsa_downloader, metrics, metrics_eval, sketch, sketch_eval` — none of
  them contain the project's signature packages (`vector_quantize_pytorch`,
  `x_transformers`, `pydiffvg`, `pytorch_lightning`). The env must be created
  from scratch.
- **conda:** `/sc/home/marco.cipriano/miniforge3` (base = py3.12 / torch 2.7.1+cu126 — unrelated).
- **GPU:** 2× Tesla T4 (16 GB), driver 570.211.01, CUDA 12.8 runtime — compute is available.
- **diffvg** is checked out + built at `/sc/home/marco.cipriano/libs/diffvg`, **but
  the build is stale**: it was compiled for **Python 3.9** (`build/*-cpython-39/`)
  and **CPU-only** (`CMakeCache.txt: DIFFVG_CUDA:BOOL=0`). It will **not** import
  into a py3.10 env and would be CPU-only anyway → **must be rebuilt**.

### 1.2 Install recipe — VERIFIED & AUTOMATED (2026-07)

**Just run the tested one-shot installer** (creates the `SVG` env, installs CUDA
torch, builds diffvg from source with GPU, installs pinned deps, self-verifies):

```bash
cd /sc/home/marco.cipriano/projects/Grimoire
bash install.sh            # or: bash install.sh SVG /sc/home/marco.cipriano/libs/diffvg
conda activate SVG
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib   # needed at runtime (torch finds libnvrtc)
```

Pinned pip deps live in **`requirements.txt`** (header explains the two things pip
can't do: the cu118 torch wheel and the from-source diffvg build). The manual
equivalent of `install.sh`, if you prefer step-by-step:

```bash
# 1. Env
conda create -y -n SVG python=3.10 && conda activate SVG
# 2. CUDA torch (NOT default PyPI — that gives CPU torch, or worse upgrades to torch 2.12)
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
# 3. Build toolchain (nvcc 11.8 + gcc/g++ 11 matched to the T4 sm_75) + system Cairo
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit -c conda-forge gcc_linux-64=11 gxx_linux-64=11 cmake ninja cairo pango
# 4. Rebuild diffvg for THIS python + CUDA (existing /sc/home/marco.cipriano/libs/diffvg is a stale py3.9/CPU build)
cd /sc/home/marco.cipriano/libs/diffvg && git submodule update --init --recursive && rm -rf build
CUDA_HOME=$CONDA_PREFIX CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \
  TORCH_CUDA_ARCH_LIST=7.5 DIFFVG_CUDA=1 python setup.py install
# 5. Pinned deps
cd /sc/home/marco.cipriano/projects/Grimoire && pip install -r requirements.txt
# 6. Verify
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python -c "import torch,pydiffvg; print('cuda',torch.cuda.is_available(),'diffvg gpu',pydiffvg.get_use_gpu())"  # both True
```

> **Do NOT use the old `requirements.yaml`** (`conda env update --file requirements.yaml`) — it
> pins a CPU torch and re-triggers the numpy-2 / torch-upgrade footguns. Use `requirements.txt`.

**Key version pins that matter (learned the hard way):** `numpy==1.26.4` (<2, or torch
2.0.1 crashes with an ABI error) · `triton==2.0.0` (torch 2.0.1's pin; deps drag it to 3.x) ·
`setuptools<80` (lightning 2.0.6 uses the removed `pkg_resources.declare_namespace`) ·
**`opencv-python-headless`** not `opencv-python` (headless node has no `libGL.so.1`).

### 1.2b Code fixes applied to run on this env (in `experiment.py`)
The uncommitted emoji WIP had a few py/torch-version incompatibilities that block training here — all fixed (small, safe):
1. **`import os`** was missing (used by the fix in #4).
2. **Stage-1 unpack** (train/val steps): the VSQ forward's 5th element was unpacked as a 2-tuple `(all_paths, pred_colors)` — only the `cnn` head returns that; the `mlp` head (figr8) returns a `scenes` list → crash. Bound as a single unused `_scenes` (the vars are unused in Stage-1). The `VSQ_layers` experiment that *does* use them is untouched.
3. **`torch.any(x, dim=(-1,-2,-3))`** (3 sites) — multi-dim `torch.any` needs torch ≥2.1; rewrote as `(x==-1).flatten(start_dim=2).any(dim=-1)` for torch 2.0.1.
4. **Hardcoded `/home/marco.cipriano/test/showcase/…svg`** in emoji validation (unwritable here) → auto-created local `./showcase/` dir.

### 1.2c Verified smoke runs (2026-07, single Tesla T4)
| Pipeline | Launcher / config | Result |
|---|---|---|
| MNIST VSQ (stage 1, cnn) | `run.py` | loss **0.348→0.271**, clean finish |
| FIGR8 VSQ (stage 1, mlp, alpha) | `run.py` | loss **0.078→0.045** |
| FIGR8 ART (stage 2) | `run_stage2.py` | val_loss **8.34→7.89** @ ~1.1 s/step |
| emoji VSQ (stage 1, VSQ_layers+hydra) | `run.py` | trains + validates; pyramid/outline/bnw losses compute |

Node caveats: it is **shared** (other users' GPUs/kernels), the `sci-aisc` mount **intermittently hangs** on metadata/large reads, and **session teardowns SIGKILL background jobs** — for reliable smoke runs, copy small data to local `/tmp`, use `num_workers: 0`, and run each stage pinned to one GPU (`CUDA_VISIBLE_DEVICES=N`, `devices: 1`). The `mnistPrecomputed` loader `list(torch.load(train.pt))`s the whole tensor (106 GB for P128_T6) and run.py+Lightning both call `setup()` → the double in-RAM load gets killed; use a `.clone()`d subset for smokes.

### 1.3 Key dependencies (true minimal surface)
python 3.10 · torch/torchvision **2.0.1/0.15.2** · pytorch-lightning 2.0.6 ·
**pydiffvg (from source)** · vector_quantize_pytorch 1.12.17 (FSQ) ·
x-transformers 1.18.0 · transformers 4.31.0 (frozen BERT text encoder) ·
kornia 0.7.0 (LAB/pyramid losses) · lpips 0.1.4 · torch-fidelity 0.3.0 (FID) ·
torchmetrics 1.0.1 (CLIPScore) · cairosvg/cairocffi · svgpathtools/svgwrite ·
opencv-python · wandb 0.15.8 (optional) · tensorboard (default logger).

### 1.4 Gotchas ⚠️
1. **CPU-vs-GPU torch ambiguity.** `requirements.yaml` conda-pins a **CPU** torch
   (`pytorch=2.0.1=py3.10_cpu_0`, `pytorch-mutex=cpu`) but its pip section pins
   `torch==2.0.1` + `triton==2.0.0` (GPU JIT). Installing as-exported likely
   yields a **CPU torch** → you must explicitly install a CUDA build (step 4).
2. **diffvg py3.9/CPU prebuild** must be rebuilt for py3.10 + CUDA (§1.1).
3. **`clip` (OpenAI) missing** from requirements; only `models/clip_draw.py`
   needs it and it's off the main train path. The CLIP *metric*
   (`experiment.py`) uses the HF/torchmetrics path instead — no OpenAI CLIP needed.
4. **Double-pinned conflicts** (torchmetrics 0.11.2 vs 1.0.1; numpy 1.24.3 vs
   1.25.2) — pip wins in `conda env update`; harmless.
5. **README says `requirements.txt`** but only `requirements.yaml` exists.
6. **Cairo system lib** needed for `cairosvg` (`utils.py`); install `cairo` via conda-forge if `import cairosvg` fails.
7. **`einx==0.1.3`** is listed but never imported — ignore.

### 1.5 External services / manual setup
- **wandb** (optional): only used with `-w/--wandb`; default logger is
  TensorBoard. Entity defaults to `aiis-chair`; run `wandb login`, or use
  `--debug` for offline. Active configs set `entity: aiis-chair`, `project: grimoire-2`.
- **HuggingFace** (required at stage-2 runtime): downloads the frozen BERT text
  encoder (`google/bert_uncased_L-12_H-512_A-8` / `bert-base-uncased`) and the
  CLIP metric model (`openai/clip-vit-base-patch16`). All public — needs internet
  or a warm `~/.cache/huggingface`. A HF token is present (`~/.hf-cli`) but the
  code reads no token (models are public).
- **`.env`** (only for regenerating the NounProject dataset): `NOUN_PROJECT_API_KEY`/`_SECRET`.

---

## 2. Checkpoints — VSQ (stage 1) & ART (stage 2)

**Root:** `/sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/{VSQ,ART}/`
(60 VSQ experiment dirs, 9 ART dirs; each holds
`checkpoints/{last.ckpt, epoch=*-step=*.ckpt}` + `wandb/`, `Samples/`, `Reconstructions/`).
Independently verified: **309 non-empty checkpoints** (237 VSQ + 72 ART).

### 2.1 How checkpoints are produced/consumed
- **Stage 1** (`run.py`): `ModelCheckpoint` → `<save_dir>/<name>/checkpoints/`
  (`save_top_k=3`, `save_last=True`, `monitor=val_loss`). `last.ckpt` is the file
  stage 2 consumes.
- **Stage 2** (`run_stage2.py:86`): loads stage-1 weights from
  `config['stage1_params']['checkpoint_path']`; writes its own ART ckpts the same way.
- `exp_params.continue_checkpoint` = optional resume path (asserted to exist).

### 2.2 Paper-final checkpoints (map README table → real path)
All under `/sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/`:

| Dataset | Stage | Experiment dir (`.../Grimoire/…`) | `last.ckpt` |
|---|---|---|---|
| FIGR8 | **VSQ** | `VSQ/figr8/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None` | 171 MB |
| FIGR8 | **ART** | `ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid` | 1.5 GB |
| FIGR8 | **ART** | `ART/figr8/ART_S4C2G256` (= active `figr8_ART.yaml` name) | 1.5 GB |
| FONTS | **VSQ** | `VSQ/VSQ_FONTS` | 141 MB |
| MNIST-BW | **VSQ** | `VSQ/VSQ_MNIST_BW_P128_T6_P20_TH0.1` | 296 MB |
| MNIST-BW | **VSQ** | `VSQ/VSQ_MNIST_BW_P128_T6_P0_TH0.2_singleGPU_lowBS` (stage-1 input for `MNIST_ART_BW.yaml`) | 296 MB |
| MNIST-BW | **VSQ** | `VSQ/VSQ_MNIST_BW_P128_T14_P20_TH0.2` (+ `_S64`) | 296 MB |
| MNIST-BW | **ART** | `ART/ART_MNIST_BW_P6T0.2` | 1.2 GB |
| MNIST-BW | **ART** | `ART/ART_MNIST_BW_P128_T6_P0_TH0.2` (= active `MNIST_ART_BW.yaml` name) | 1.2 GB |
| MNIST-color | **VSQ** | `VSQ/VSQ_MNIST_COLOR_P128_T3_P20` | 296 MB |
| MNIST-color | **ART** | `ART/MNIST_COLOR` (+ `MNIST_COLOR/bnw/`) | 1.2 GB |
| Cartoon | **VSQ** | `VSQ/VSQ_Cartoon_COLOR_with_eyes` | 296 MB |
| Emoji | **VSQ** | `VSQ/VSQ_EMOJI_COLOR` | 296 MB |

Sizes are diagnostic: **VSQ tokenizers ≈ 141–310 MB**, **ART transformers ≈ 1.2–1.5 GB**.

Plus many sweep/ablation VSQ dirs (`VSQ/{1..12}_P*_T*`, the full emoji/`hydra`
family `emoji_VSQ_P{128,256}_S{32,64}_C{4096,8192}_d{512,1024}[_hydra…]`) and ART
ablations (`ART/ART_MNIST_BW_{AUG,P6,P14,P14S64}`, `ART/figr8/nseg=*` variants).

### 2.3 Gaps (not recoverable)
- `MNIST_ART.yaml` references stage-1 `VSQ/TiledMNIST/...` — **no such dir**; the
  color-MNIST tokenizer actually used is `VSQ_MNIST_COLOR_P128_T3_P20` (present).
- **FONTS-ART** and **MNIST-im2vec** were never produced (blank in README; no dir).

---

## 3. Datasets & HuggingFace

**Root:** `/sc/projects/sci-aisc/marco.cipriano/data/SVG/Grimoire/`
(same 1:1 swap from the README's `/raid/.../data/SVG/Grimoire/`).

| Dataset | On disk (verified) | Size | How generated | HuggingFace |
|---|---|---|---|---|
| **MNIST** (SVG-tiled) | `MNIST/{mnist_png, mnist_pretiled, mnist_tokenized}` | large (~TB across all P/T/TH variants) | `datasets/make_font/download.sh` → `datasets/setup.py` → `scripts/create_precomputed_mnist.py` / `make_causal_positional_dataset.py` | ❌ not published |
| **FIGR-8** | `figr8/{svgs_simplified, deepsvg, tokenized/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None/}` (`split.csv` ~77 MB, `tokenized.npy` ~455 MB) | — | `datasets/convert_figr8.py` + deepsvg preprocess + `make_causal_positional_dataset.py` | ⚠️ **backup only** — `potpov/figr8_bu` (27 GB zip, cached); refs-only `potpov/figr8`, `potpov/figr8_ours`. Collaborator's private namespace, not a release |
| **Fonts** | `fonts/{svgs_simplified/{split.csv 879 MB, split_fixed.csv 348 MB}, tokenized}` — **only these two subdirs** (no `glyphazzn/allfreefonts/dafont/filtered.csv`) | ~1.2 GB CSVs | `datasets/make_font/{download_glyphazzn_dataset.py, scrape_fonts.py, ttf_to_svg.py}` + deepsvg. Needs external Glyphazzn download (many URLs now 404) | ⚠️ **backup only** — `potpov/fonts_bu` (refs cached, blobs not pulled) |
| **Emoji** (Twemoji) | `emoji_selection/{raw, preprocessed, preprocessed_v2}` | 8.7 GB | `datasets/make_emoji_dataset.py` (clone `twitter/twemoji`, set `PATH_TO_TWEMOJI`) + SAM masks | ❌ (Twemoji is external open-source) |
| **Cartoons** (cartoonset10k) | `Cartoons/{raw, preprocessed}` | 534 GB | `datasets/make_cartoon_dataset.py` (download cartoonset10k + SAM `sam_vit_h_4b8939.pth`) | ❌ (Google cartoonset is external) |
| **MNIST++** | ❌ not found | — | `datasets/setup.py::make_mnist_pp()` | ❌ |
| **TheNounProject** | ❌ not found (scratch dirs gone) | — | `datasets/make_nounproject_dataset.py` (paid Noun Project API) | ❌ — README TODO, effectively unrecoverable |

### 3.1 HuggingFace verdict
**The Grimoire datasets and checkpoints are NOT published on HuggingFace as a
citable release.** In the code, `from_pretrained` is only ever the **pretrained
BERT text encoder** and the **CLIP metric model** — never the project's data
(no `load_dataset`/`hf_hub_download`/`snapshot_download`/`repo_id` for datasets).
The only Grimoire-relevant HF artifacts are **private backups under a
collaborator's account `potpov`** (`figr8_bu`, `fonts_bu`). The `SVGsquad/*` and
`sketchingsquad/*` datasets in the HF cache belong to a **different, later
project** — do not conflate them.

### 3.2 Config path caveats
- All active configs point at the dead `/raid/...` (or one at `/home/...`) —
  repoint to `sci-aisc` (§0).
- `configs/VSQ_sweep/*` + `Base_VSQ_config.yaml` use bare relative
  `csv_path: datasets/figr8_final_paper_split.csv`, but the file is actually at
  **`datasets/make_font/figr8_final_paper_split.csv`** — fix or symlink.
- README's `datasets/download.sh`, `ttf_to_svg.py`, `download_glyphazzn_dataset.py`
  actually live under **`datasets/make_font/`**.

---

## 4. Launchers & how to run

### 4.1 Reference
| Script | Stage / purpose | Model → Experiment | Flags | Needs |
|---|---|---|---|---|
| `run.py` | **Stage 1 (VSQ)** train (+ legacy VAE/VectorGPT) | `VSQ` → `VectorVQVAE_Experiment_Stage1` | `-c/--config`, `-w/--wandb`, `--debug` | `model_params/data_params/exp_params/trainer_params/logging_params` |
| `run_stage2.py` | **Stage 2 (ART)** train | `VQ_SVG_Stage2` (+ frozen VSQ) → `SVG_VQVAE_Stage2_Experiment` | `-c`, `-w`, `--debug` | **+ `stage1_params.checkpoint_path`** |
| `run_raster_stage2.py` | Stage 2, **raster** tokenizer variant | `VQ_SVG_Stage2` (`RasterVQTokenizer`) | `-c`, `-w`, `-w_id`, `--debug` | no ready-to-run active config in repo |
| `run_sweep.py` | Legacy wandb sweep (VectorGPT only) | — | none (hardcoded) | base config path is stale; not part of VSQ→ART pipeline |

Notes: default logger = TensorBoard (unless `-w`); `--debug` → `num_workers=0` +
wandb offline; stage 2 uses DDP `ddp_find_unused_parameters_true`; `devices: -1`
(all GPUs); GPU required (diffvg).

### 4.2 Config → launcher
- **`run.py` (stage 1, VSQ):** `configs/MNIST/MNIST_VSQ*.yaml`,
  `configs/fonts/fonts_VSQ.yaml`, `configs/cartoons/*VSQ*.yaml`,
  `configs/VSQ_sweep/*`, `configs/mnist_sweep/*`.
  (`emoji_VSQ.yaml` uses `VSQ_layers` → `…Layer_Stage1`.)
- **`run_stage2.py` (stage 2, ART):** `configs/MNIST/MNIST_ART*.yaml`,
  `configs/figr8/figr8_ART.yaml`. *(No stage-2 config ships for fonts/emoji/cartoons.)*
- ⚠️ `configs/VSQ_sweep/*` set `dataset: "stage1"` which is **not** in `run.py`'s
  `DATASETMAP` → they fail an assertion until the `dataset` key is fixed.

### 4.3 End-to-end commands (MNIST black&white — primary path)
Assumes the `/raid → sci-aisc` symlink from §0 (otherwise edit each `<FILL IN>`).

```bash
conda activate SVG

# --- STAGE 1: train VSQ tokenizer ---
python run.py -c configs/MNIST/MNIST_VSQ_BW.yaml            # -w wandb, --debug for workers=0/offline
# → <save_dir>/VSQ_MNIST_BW_.../checkpoints/last.ckpt

# --- between stages: tokenize the raster dataset with the trained VSQ ---
# edit scripts/tokenize_raster_dataset.py hardcoded paths (MODEL_WEIGHTS_PATH/OUT_PATH/CONFIG_PATH)
python scripts/tokenize_raster_dataset.py                  # → vsq_tokenized.npy, text_tokenized.npy, split.csv

# --- STAGE 2: train ART transformer ---
# in configs/MNIST/MNIST_ART_BW.yaml set: stage1_params.checkpoint_path,
#   data_params.csv_path, data_params.vq_token_npy_path, logging_params.save_dir
python run_stage2.py -c configs/MNIST/MNIST_ART_BW.yaml     # -w wandb
```

**To just USE the released checkpoints** (skip training), point the stage-2
config's `stage1_params.checkpoint_path` at the VSQ `last.ckpt` from §2.2 and run
inference (§4.4). E.g. FIGR8 stage-2 generation uses
`ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid/checkpoints/last.ckpt` on top of
`VSQ/figr8/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None/checkpoints/last.ckpt`.

FIGR8 / color-MNIST: swap in `configs/figr8/figr8_ART.yaml` /
`configs/MNIST/MNIST_ART.yaml`. Emoji/Cartoons ship **stage-1 only**
(`configs/cartoons/*`); a stage-2 config must be authored (copy `figr8_ART.yaml`).

### 4.4 Inference & evaluation
| Entry point | Role |
|---|---|
| `eval.py --config <eval.yaml>` | Main benchmark harness. `type: stage1/vsq` → VSQ reconstruction FID/CLIP/MSE; `type: stage2` → ART generation + svg-fixing + FID/CLIP. **No eval yaml ships** — must be authored (needs `type/config_path/ckpt_path/out_base_dir/dataset` + stage-2 sampling keys). |
| `scripts/run_stage_1_inference.py --checkpoint_path … --dataset_csv_path … --outpath …` | Reconstruct one SVG with a VSQ ckpt (`original.svg` + `reconstructed.svg`). |
| `scripts/tokenize_svg_dataset.py` | SVG→VQ-token preprocessing for stage 2 (fonts/figr8); hardcoded paths. |
| `scripts/tokenize_raster_dataset.py` | Raster→VQ-token preprocessing for stage 2 (MNIST); hardcoded paths. |
| `scripts/create_precomputed_mnist.py` | Pre-tile MNIST PNGs into patch tensors for `mnistPrecomputed`. |
| `scripts/direct_optimization.py` + `run_direct_optimization.sh` | Baseline: direct diffvg path optimization toward a target image. |
| `scripts/svg2gif.py` / `svg2pdf.py` | Render generated SVGs → progressive GIFs / paper PDFs. |
| `notebooks/inference.ipynb`, `inference_raster.ipynb`, `visualizations.ipynb` | Interactive stage-1/2 generation and paper figures. |
| `scripts/run_stage_2_inference.py`, `scripts/stage2_forward.py` | ⚠️ stale — reference removed `VQ_Transformer` / old `/scratch2` paths. |

---

## 5. Architecture cheat-sheet

**Two-stage conditional SVG generator.**
- **Stage 1 = VSQ** (`models/vsq.py`, class `VSQ`): a vector-quantized autoencoder
  over SVG shape patches. ResNet-18 encoder → **FSQ** discrete bottleneck
  (`vector_quantize_pytorch`; codebook size = `∏ fsq_levels`, so the config's
  `codebook_size` is ignored) → vector decoder head (`mlp`/`cnn`/`hydra`) emitting
  Bézier params → DiffVG renders back to raster for the loss. It's the SVG **tokenizer**.
- **Stage 2 = ART** (`models/stage2.py`, class `VQ_SVG_Stage2`): a causal
  `x_transformers` decoder over sequences of *(VSQ code, position)* tokens,
  conditioned on a frozen **BERT** text embedding. Sampling from text → token
  sequence → VSQ decoder → full SVG.
- **Bridge = `VQTokenizer`** (`tokenizer.py`): uses the frozen stage-1 VSQ to map
  patches → codebook indices, adds position + special tokens, detokenizes back to SVG.
- "**ART**" is only a name in result paths / README — the code class is `VQ_SVG_Stage2`.

**Config sections:** `model_params` (net being trained), `data_params` (dataset +
paths + patch/grid sizes), `exp_params` (lr/seed/`continue_checkpoint`),
`trainer_params` (`devices`, `max_epochs`), `logging_params`
(`save_dir`/`name`/`project`/`author`/wandb `entity`), and **stage-2 only**
`stage1_params` (frozen VSQ spec + `checkpoint_path`).

**Config-name shorthand:** `P`=patch_size · `S`/`T`=tiles(segments) per row ·
`C`=codebook · `d`=quantized_dim · `nseg`=num_segments · `ncode`=num_codes_per_shape ·
`lseg`=segment length · `alpha`=geometric-constraint weight · `TH`=threshold.

---

## 6. Quick verification commands

```bash
# checkpoints present?
ls /sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/VSQ | head
ls /sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/ART
# a specific last.ckpt
ls -lh "/sc/projects/sci-aisc/marco.cipriano/results/svg/Grimoire/ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid/checkpoints/last.ckpt"
# datasets present?
ls /sc/projects/sci-aisc/marco.cipriano/data/SVG/Grimoire
# env / diffvg after install
conda activate SVG && python -c "import torch, pydiffvg; print('cuda', torch.cuda.is_available(), 'diffvg gpu', pydiffvg.get_use_gpu())"
```

---

## 7. HuggingFace artifacts & inference (published 2026-07, account `Potpov`, public)

**Checkpoints** — [Potpov/grimoire-checkpoints](https://huggingface.co/Potpov/grimoire-checkpoints)
(model repo). All 9 final checkpoints, each with its training `config.yaml`, under
`<dataset>/<stage>/{last.ckpt,config.yaml}`: `figr8/{vsq,art}`, `mnist_bw/{vsq,art}`,
`mnist_color/{vsq,art}`, `fonts/vsq`, `emoji/vsq`, `cartoon/vsq`.

**Datasets** (per "processed if regen is painful, raw if easy"):
- [Potpov/grimoire-figr8](https://huggingface.co/datasets/Potpov/grimoire-figr8) — **processed**: `tokenized/{tokenized.npy,split.csv}` (ART-ready) + `svgs_simplified.tar.gz` (VSQ corpus).
- [Potpov/grimoire-mnist](https://huggingface.co/datasets/Potpov/grimoire-mnist) — **raw** `mnist_png.tar.gz` (regen to tensors via `create_precomputed_mnist.py`, verified).
- [Potpov/grimoire-emoji](https://huggingface.co/datasets/Potpov/grimoire-emoji) — **processed** `preprocessed_v2.tar` masks (+ `raw.tar.gz`); regen needs SAM.
- Skipped: `mnist_pretiled` (2.7 TB, regenerable), Cartoons (534 GB), fonts (21 GB), MNIST++ (absent).

**Inference from HF — both stages verified.** Two working scripts (added this session):
```bash
# Stage 1 (VSQ) — download checkpoint from HF, rebuild from bundled config, reconstruct one SVG
python scripts/hf_inference_demo.py --hf_repo Potpov/grimoire-checkpoints --subdir figr8/vsq \
    --dataset_csv_path <csv with a file_path column> --outpath out/
# Stage 2 (ART) — load VSQ+ART from HF, generate an SVG from text
python scripts/hf_generate_demo.py --hf_repo Potpov/grimoire-checkpoints \
    --art_subdir figr8/art --vsq_subdir figr8/vsq --prompt "a star" --outfile out/gen.svg
```
Both confirmed: VSQ loads with 0 missing/unexpected keys and reconstructs; ART loads + generates
a valid SVG. Note: build the VSQ from its **own** `vsq/config.yaml` — the ART config's
`stage1_params` under-specifies the VSQ architecture (encoder fc 512 vs 1024). The legacy
`scripts/run_stage_{1,2}_inference.py` are superseded by these (they built VSQ without the
architecture params / referenced removed classes).
