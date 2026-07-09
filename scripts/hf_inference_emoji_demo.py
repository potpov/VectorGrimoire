"""
Emoji / layered-color inference: load the emoji VSQ (VSQ_layers + hydra) checkpoint from
HuggingFace and reconstruct one emoji as a multi-layer COLORED SVG.

This is the layered path (distinct from the single-layer figr8/fonts reconstruction in
hf_inference_demo.py): it runs the VSQ forward, then `layer_recon` + per-layer fill colors,
exactly like the VSQ_layers validation step.

Example:
    python scripts/hf_inference_emoji_demo.py --emoji_dir <dir containing preprocessed_v2/> \
        --outfile out/emoji_recon.svg
"""
import os
import sys
import argparse
import yaml
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import hf_hub_download
import pydiffvg
from models import VSQ
from dataset import CartoonDataset
from utils import layer_recon


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(hf_hub_download(args.hf_repo, f"{args.subdir}/config.yaml")))
    ckpt = hf_hub_download(args.hf_repo, f"{args.subdir}/last.ckpt")
    mp = {k: v for k, v in cfg["model_params"].items() if k != "name"}
    print(f"[hf] emoji VSQ: decoder={mp.get('vector_decoder_model')}, num_segments={mp.get('num_segments')}, "
          f"num_codes={mp.get('num_codes_per_shape')}, pred_color={mp.get('pred_color')}")
    model = VSQ(**mp, device=device)
    sd = torch.load(ckpt, map_location=device)["state_dict"]
    try:
        model.load_state_dict(sd)
    except Exception:
        model.load_state_dict({k.replace("model.", ""): v for k, v in sd.items()}, strict=False)
    model = model.to(device).eval()
    print("[load] emoji VSQ loaded from HF")

    data = CartoonDataset(data_path=args.emoji_dir, layer_length=args.layer_length,
                          train_batch_size=1, val_batch_size=1, num_workers=0,
                          patch_size=args.patch_size, return_raw=True)
    data.setup()
    batch = next(iter(data.val_dataloader()))
    patches, labels, centers, descriptions, color_weights, gt_outlines, gt_bnw, raw_imgs = batch
    bs, cl, c, w, h = patches.shape
    layer_att_mask = (~(patches == -1).flatten(start_dim=2).any(dim=-1)).reshape(bs * cl)
    patches = patches.reshape(bs * cl, c, w, h).to(device)

    with torch.no_grad():  # not inference_mode: layer_recon does in-place edits on the outputs
        out, _ = model.forward(patches)
    reconstructions, _, all_points, vq_loss, (all_paths, pred_colors), pred_outline, pred_bnw = out
    print(f"[infer] VSQ_layers forward OK — {len(all_paths)} decoded shapes across {cl} layers")

    real_paths, colors = layer_recon(layer_att_mask, cl, all_paths, pred_colors, centers, 0, patch_size=w)
    groups = [pydiffvg.ShapeGroup(shape_ids=torch.tensor([j]), fill_color=colors[j], stroke_color=colors[j])
              for j in range(len(colors))]
    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    pydiffvg.save_svg(args.outfile, 400, 400, real_paths, groups)
    print(f"[done] wrote colored layered reconstruction -> {args.outfile} ({len(colors)} colored layers)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Emoji layered/colored reconstruction from an HF VSQ checkpoint")
    p.add_argument("--hf_repo", default="Potpov/grimoire-checkpoints")
    p.add_argument("--subdir", default="emoji/vsq")
    p.add_argument("--emoji_dir", required=True, help="dir containing preprocessed_v2/ (from Potpov/grimoire-emoji)")
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--layer_length", type=int, default=16)
    p.add_argument("--outfile", default="scripts/output/emoji_recon.svg")
    main(p.parse_args())
    print("Finished successfully.")
