"""
Stage-2 (ART) text -> SVG generation, loading BOTH checkpoints straight from HuggingFace.

Downloads <art_subdir>/{last.ckpt,config.yaml} and <vsq_subdir>/last.ckpt from the HF
model repo, rebuilds the frozen VSQ tokenizer + the autoregressive VQ_SVG_Stage2 model
(architecture taken from the ART run's config), loads both state dicts, and samples an
SVG from a text prompt.

Example:
    python scripts/hf_generate_demo.py --prompt "a star" --outfile out/gen.svg
"""
import os
import sys
import argparse
import yaml
import torch
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import VSQ, VQ_SVG_Stage2
from tokenizer import VQTokenizer


def _remap_key(k):
    # Strip ONLY the leading Lightning "model." wrapper — a blanket k.replace("model.","")
    # also deletes the inner ".model." in "transformer.model.layers.*", so the whole
    # transformer silently fails to load and runs on random weights.
    if k.startswith("model."):
        k = k[len("model."):]
    # x-transformers version shift: the FF output linear moved from index .ff.3 to .ff.2.
    return k.replace(".ff.3.", ".ff.2.")


def _load_sd(path, model, device, expect_prefixes=()):
    sd = torch.load(path, map_location=device)["state_dict"]
    fixed = {_remap_key(k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(fixed, strict=False)
    core_missing = [k for k in missing if k.startswith(expect_prefixes)] if expect_prefixes else []
    if core_missing:
        print(f"[warn] {len(core_missing)} expected weights did not load, e.g. {core_missing[:3]}")
    return model


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = args.hf_repo
    art_cfg = yaml.safe_load(open(hf_hub_download(repo, f"{args.art_subdir}/config.yaml")))
    art_ckpt = hf_hub_download(repo, f"{args.art_subdir}/last.ckpt")
    vsq_cfg = yaml.safe_load(open(hf_hub_download(repo, f"{args.vsq_subdir}/config.yaml")))
    vsq_ckpt = hf_hub_download(repo, f"{args.vsq_subdir}/last.ckpt")
    print(f"[hf] downloaded ART + VSQ checkpoints/configs from {repo}")

    dp = art_cfg["data_params"]; mp = dict(art_cfg["model_params"]); mp.pop("name", None)
    # build the VSQ from ITS OWN config (authoritative for the stage-1 architecture)
    vsq_mp = {k: v for k, v in vsq_cfg["model_params"].items() if k != "name"}
    num_codes = vsq_mp["num_codes_per_shape"]
    lseg = vsq_cfg.get("data_params", {}).get("individual_max_length",
                                              art_cfg["stage1_params"].get("lseg", 5.0))

    # 1) frozen VSQ + tokenizer
    vq_model = VSQ(**vsq_mp, device=device)
    vq_model = _load_sd(vsq_ckpt, vq_model, device).to(device).eval()
    tokenizer = VQTokenizer(vq_model, dp["grid_size"], num_codes,
                            mp["text_encoder_str"], lseg=lseg, device=device)
    print(f"[build] VSQ tokenizer ready (num_codes={num_codes}, lseg={lseg})")

    # 2) autoregressive ART transformer + load trained weights
    model = VQ_SVG_Stage2(tokenizer, **mp, device=device)
    model = _load_sd(art_ckpt, model, device).to(device).eval()
    print("[build] VQ_SVG_Stage2 (ART) ready; generating...")

    with torch.inference_mode():
        drawing = model._generate_from_text(
            args.prompt, temperature=args.temperature,
            sampling_method="top_p", sampling_kwargs={"thres": 0.9},
            return_drawing=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    drawing.saveas(args.outfile)
    print(f"[done] generated SVG for '{args.prompt}' -> {args.outfile}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Grimoire stage-2 text->SVG generation from HF checkpoints")
    p.add_argument("--hf_repo", default="Potpov/grimoire-checkpoints")
    p.add_argument("--art_subdir", default="figr8/art")
    p.add_argument("--vsq_subdir", default="figr8/vsq")
    p.add_argument("--prompt", default="a star")
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--outfile", default="scripts/output/generated.svg")
    main(p.parse_args())
    print("Finished successfully.")
