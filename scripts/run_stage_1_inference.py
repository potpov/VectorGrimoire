# ============================================================================
# DEPRECATED / limited — builds VSQ with only (codebook_size, vq_method), so it
# cannot correctly load real checkpoints (which need vector_decoder_model,
# num_segments, num_codes_per_shape, ...), and its default paths are dead.
# For stage-1 (VSQ) reconstruction from a real/HF checkpoint use
# scripts/hf_inference_demo.py (figr8/fonts) or scripts/hf_inference_emoji_demo.py
# (emoji, layered+colored). Kept for historical reference.
# ============================================================================
import os
import torch
from models import VSQ
from dataset import CenterShapeLayersFromSVGDataset
from utils import calculate_global_positions, shapes_to_drawing
import argparse
import random

def main(args):
    # Your existing code here
    
    ds = CenterShapeLayersFromSVGDataset(args.dataset_csv_path, 
                                     channels=3, 
                                     width=128, 
                                     train=args.use_train_set, 
                                     stroke_width=0.7,
                                     individual_max_length=7.5,
                                     max_shapes_per_svg=300)
    
    model = VSQ(codebook_size=args.codebook_size, vq_method=args.vq_method)
    state_dict = torch.load(args.checkpoint_path)["state_dict"]
    # map keys to remove the prefix "model."
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.eval()

    idx = random.choice(range(len(ds)))
    shape_layers, _, positions = ds.__getitem__(idx)

    with torch.inference_mode():
        out, _ = model.forward(shape_layers)
    
    original_drawing = ds._get_full_svg_drawing(3)
    global_shapes = calculate_global_positions(out[2], ds.individual_max_length + 2, positions)[:,0]  # +2 to account for padding in the rasterization process
    reconstructed_drawing = shapes_to_drawing(global_shapes, stroke_width=ds.stroke_width, w=72.)

    original_drawing.filename = os.path.join(args.outpath, "original.svg")
    reconstructed_drawing.filename = os.path.join(args.outpath, "reconstructed.svg")
    original_drawing.save()
    reconstructed_drawing.save()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run stage 1 inference script")
    parser.add_argument("--dataset_csv_path", type=str, help="Path to CSV file", default="/scratch2/moritz_data/glyphazzn/B_simplified/split.csv")
    parser.add_argument("--use_train_set", type=bool, help="Use training set samples", default=False)
    parser.add_argument("--stroke_width", type=float, default=0.7, help="Stroke width")
    parser.add_argument("--individual_max_length", type=float, default=7.5, help="Individual max length of a stroke")
    parser.add_argument("--codebook_size", type=int, default=128, help="Codebook size")
    parser.add_argument("--vq_method", type=str, default="fsq", help="VQ method")
    parser.add_argument("--checkpoint_path", type=str, help="Path to checkpoint file", default="/scratch2/moritz_logs/SVG_VQVAE/Stage1/glyphazzn_B_simplified_single_code/checkpoints/last-v3.ckpt")
    parser.add_argument("--outpath", type=str, help="Path to save the svg files", default="/home/mfeuerpfeil/master/thesis/scripts/output")
    
    args = parser.parse_args()
    main(args)
    print("Finished successfully.")
    