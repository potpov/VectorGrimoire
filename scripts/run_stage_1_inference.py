import os
import torch
from torch import Tensor
from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import glob
import pandas as pd
import numpy as np
from models import Vector_VQVAE

from utils import svg2paths2, disvg, raster, get_single_paths, get_similar_length_paths, check_for_continouity, get_rasterized_segments, all_paths_to_max_diff
import copy
import string
from dataset import CenterShapeLayersFromSVGDataset
from svgpathtools import svg2paths, svg2paths2, disvg, Path, CubicBezier
from svgwrite import Drawing
from thesis.utils import calculate_global_positions, shapes_to_drawing
import argparse

def main(args):
    # Your existing code here
    
    ds = CenterShapeLayersFromSVGDataset(args.dataset_csv_path, 
                                     channels=3, 
                                     width=128, 
                                     train=args.use_train_set, 
                                     stroke_width=0.7,
                                     individual_max_length=7.5,
                                     max_shapes_per_svg=300)
    
    model = Vector_VQVAE(codebook_size=args.codebook_size, vq_method=args.vq_method)
    state_dict = torch.load(args.checkpoint_path)["state_dict"]
    # map keys to remove the prefix "model."
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.eval()

    shape_layers, _, positions = ds.__getitem__(3)

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
    