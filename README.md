# 🧙 Vector Grimoire: Codebook-based Shape Generation under Raster Image Supervision

This is the official repository of Vector Grimoire, published at ICML 2025. Vector Grimoire is a two-stage, text-conditional SVG generative model.

- **VSQ** (stage 1) — a vector-quantized SVG tokenizer/autoencoder: ResNet encoder → FSQ codebook → differentiable vector decoder rendered with [diffvg](https://github.com/BachiLi/diffvg).
- **ART** (stage 2) — an autoregressive transformer over VSQ tokens, conditioned on a frozen BERT text embedding (class `VQ_SVG_Stage2`). Turns a text prompt into an SVG.

Pretrained checkpoints and datasets are on the Hugging Face Hub under [`Potpov`](https://huggingface.co/Potpov).

---

## 1. Installation

Verified with **Python 3.10 · PyTorch 2.0.1 (CUDA 11.8) · diffvg built from source** on NVIDIA Tesla T4 (sm_75).

```bash
conda create -n SVG python=3.10 && conda activate SVG
bash install.sh                 # torch cu118 + CUDA toolkit + diffvg-from-source + pip deps
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib   # needed at runtime so torch finds libnvrtc
```

Two components are **not** on PyPI and are installed (in this order) by `install.sh` before `pip install -r requirements.txt`:

1. **PyTorch 2.0.1 / cu118** — `pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118` (keep `numpy<2`; torch 2.0.1's ABI breaks on numpy 2.x).
2. **diffvg (pydiffvg)** — built from source against this env's Python + CUDA (`TORCH_CUDA_ARCH_LIST=7.5`, `DIFFVG_CUDA=1`).

Full step-by-step (troubleshooting, exact pins, non-T4 GPUs) → [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md).

---

## 2. Checkpoints, training & inference

All checkpoints live in one HF model repo — **[`Potpov/grimoire-checkpoints`](https://huggingface.co/Potpov/grimoire-checkpoints)** — laid out as `<dataset>/<stage>/{last.ckpt,config.yaml}`.

| Dataset       | VSQ (stage 1) | ART (stage 2) | Train config |
|---------------|:-:|:-:|:--|
| **figr8**     | [figr8/vsq](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/figr8/vsq) | [figr8/art](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/figr8/art) | `configs/figr8/figr8_ART.yaml` |
| **mnist (b/w)** | [mnist_bw/vsq](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/mnist_bw/vsq) | [mnist_bw/art](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/mnist_bw/art) | `configs/MNIST/MNIST_{VSQ,ART}_BW.yaml` |
| **mnist (color)** | [mnist_color/vsq](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/mnist_color/vsq) | [mnist_color/art](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/mnist_color/art) | `configs/MNIST/MNIST_{VSQ,ART}.yaml` |
| **fonts**     | [fonts/vsq](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/fonts/vsq) | — | `configs/fonts/` |
| **emoji**     | [emoji/vsq](https://huggingface.co/Potpov/grimoire-checkpoints/tree/main/emoji/vsq) | — | `configs/cartoons/emoji_VSQ.yaml` (layered/colored VSQ) |

> ART was only deployed on **figr8** and **mnist**. Fonts/emoji ship the VSQ stage only.

### Train

```bash
# Stage 1 — VSQ  (all datasets)
python run.py -c configs/MNIST/MNIST_VSQ_BW.yaml

# Stage 2 — ART  (svg-tokenized datasets: figr8, fonts)
python run_stage2.py -c configs/figr8/figr8_ART.yaml

# Stage 2 — ART  (raster-tokenized datasets: mnist)
python run_raster_stage2.py -c configs/MNIST/MNIST_ART_BW.yaml
```

Point one GPU at a run with `CUDA_VISIBLE_DEVICES=<id>` and `devices: 1` in the config. Set `wandb: false` (or override the `entity`) to skip Weights & Biases.

### Inference (loads checkpoints straight from HF)

```bash
# VSQ reconstruction  (figr8 / fonts — single-layer)
python scripts/hf_inference_demo.py --subdir figr8/vsq \
    --dataset_csv_path <csv with a file_path column> --outpath out/

# ART text -> SVG  (loads VSQ + ART, samples an SVG)
python scripts/hf_generate_demo.py --art_subdir figr8/art --vsq_subdir figr8/vsq \
    --prompt "a star" --outfile out/gen.svg

# VSQ reconstruction  (emoji — layered, colored)
python scripts/hf_inference_emoji_demo.py --subdir emoji/vsq \
    --emoji_dir <dir with preprocessed_v2/> --outfile out/emoji.svg
```

All three default to `--hf_repo Potpov/grimoire-checkpoints`. See the [checkpoints repo README](https://huggingface.co/Potpov/grimoire-checkpoints) for the state-dict loading notes (leading-`model.`-prefix strip; ART also needs a `.ff.3`→`.ff.2` key remap, handled by `hf_generate_demo.py`).

---

## 3. Datasets

| Dataset  | Hugging Face | Notes |
|----------|--------------|-------|
| **figr8** | [Potpov/grimoire-figr8](https://huggingface.co/datasets/Potpov/grimoire-figr8) | Simplified SVGs + tokenized versions. Our split differs from the original FIGR-8 paper — see the dataset card. |
| **mnist** | [Potpov/grimoire-mnist](https://huggingface.co/datasets/Potpov/grimoire-mnist) | Rasterized MNIST + pre-tiled/pre-tokenized VSQ variants (b/w and color). |
| **emoji** | [Potpov/grimoire-emoji](https://huggingface.co/datasets/Potpov/grimoire-emoji) | Preprocessed open-source Twitter emojis (`preprocessed_v2/`). |
| **fonts** (Glyphazzn) | — | Not redistributable (source-font licensing). Rebuild locally; a VSQ checkpoint is provided. |

Each tokenized variant is keyed by its VSQ hyperparameters (patch size, tokens/patch, positions, threshold). Use the tokenized version whose key matches the checkpoint you load.

---

## Citation

If you use our work, please cite us:

```bibtex
@inproceedings{cipriano2025vectorgrimoire,
  title     = {Vector Grimoire: Codebook-based Shape Generation under Raster Image Supervision},
  author    = {Cipriano, Marco and Feuerpfeil, Moritz and De Melo, Gerard},
  booktitle = {Forty-second International Conference on Machine Learning},
  year      = {2025},
  month     = {October},
}
```
