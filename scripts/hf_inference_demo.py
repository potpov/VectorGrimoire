"""
Stage-1 (VSQ) reconstruction demo, loading a Grimoire VSQ checkpoint straight from
HuggingFace. Downloads <subdir>/{last.ckpt,config.yaml}, rebuilds the VSQ from the
bundled config, then reconstructs a few SVGs using the model's own `reconstruct()`
(the same assembler the training/validation logging uses) and writes
{i}_original.svg + {i}_reconstructed.svg.

Works for the SVG-based tokenizers (figr8 / fonts). For the layered emoji tokenizer
use scripts/hf_inference_emoji_demo.py instead.

Example:
    python scripts/hf_inference_demo.py --hf_repo Potpov/grimoire-checkpoints --subdir figr8/vsq \
        --dataset_csv_path <csv with a `simplified_svg_file_path` column> --outpath out/ --num 4
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import hf_hub_download
from models import VSQ
from dataset import VSQDataset


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(hf_hub_download(args.hf_repo, f"{args.subdir}/config.yaml")))
    mp = {k: v for k, v in cfg["model_params"].items() if k != "name"}
    dp = cfg.get("data_params", {})
    print(f"[hf] building VSQ: decoder={mp.get('vector_decoder_model')}, num_segments={mp.get('num_segments')}, "
          f"num_codes={mp.get('num_codes_per_shape')}")
    model = VSQ(**mp, device=device)
    sd = torch.load(hf_hub_download(args.hf_repo, f"{args.subdir}/last.ckpt"), map_location=device)["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()})
    model = model.to(device).eval()
    print("[load] VSQ loaded from HF")

    ds = VSQDataset(
        args.dataset_csv_path,
        channels=dp.get("channels", 3), width=dp.get("width", 128), train=args.use_train_set,
        stroke_width=dp.get("stroke_width", 0.56),
        individual_max_length=dp.get("individual_max_length", 5.0),
        max_shapes_per_svg=dp.get("max_shapes_per_svg", 32),
    )
    assert len(ds) > 0, "dataset is empty — check the csv `split` column (use --use_train_set)"
    os.makedirs(args.outpath, exist_ok=True)
    for idx in range(min(args.num, len(ds))):
        patches, labels, positions, _ = ds._get_full_item(idx)
        with torch.no_grad():
            recon, _ = model.reconstruct(patches.to(device), positions.to(device),
                                         ds.individual_max_length + 2,
                                         local_stroke_width=args.stroke_width, rendered_w=args.width)
        ds._get_full_svg_drawing(idx, width=args.width).saveas(os.path.join(args.outpath, f"{idx}_original.svg"))
        recon.saveas(os.path.join(args.outpath, f"{idx}_reconstructed.svg"))
        print(f"[{idx}] reconstructed {len(patches)} shapes -> {args.outpath}/{idx}_reconstructed.svg")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Grimoire VSQ (stage-1) reconstruction from an HF checkpoint")
    p.add_argument("--hf_repo", default="Potpov/grimoire-checkpoints")
    p.add_argument("--subdir", default="figr8/vsq")
    p.add_argument("--dataset_csv_path", required=True, help="CSV with a `simplified_svg_file_path` column")
    p.add_argument("--use_train_set", action="store_true", help="use the train split (default: val)")
    p.add_argument("--num", type=int, default=4, help="number of samples to reconstruct")
    p.add_argument("--stroke_width", type=float, default=0.04)
    p.add_argument("--width", type=float, default=480.0)
    p.add_argument("--outpath", default="scripts/output")
    main(p.parse_args())
    print("Finished successfully.")
