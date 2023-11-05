import pandas as pd
import numpy as np
import os
import yaml
import torch
from models import VectorGPTv2
from dataset import NewCausalSVGDataModule
from glob import glob
import random
from datasets.make_causal_positional_dataset import all_paths_to_max_diff, get_single_paths, svg2paths2, get_positional_array_from_paths
from svgpathtools import Path, Line, CubicBezier, disvg

print(torch.cuda.is_available())

SKIP_WEIGHT_LOADING = False


config_file_path = "/home/mfeuerpfeil/master/thesis/configs/VectorGPTv2_overfit_B.yaml"
with open(config_file_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

model = VectorGPTv2(**config['model_params'])

if not SKIP_WEIGHT_LOADING:
    ckpt_path = "/scratch2/moritz_logs/VectorGPTv2/google_font/merged_input/checkpoints/last-v1.ckpt"
    state_dict = torch.load(ckpt_path)["state_dict"]

    mapped_state_dict = {}
    for key in state_dict.keys():
        mapped_state_dict[key.replace("model.", "")] = state_dict[key]

    model.load_state_dict(state_dict, strict=False)
model.eval()
print()

def line_to_cubic(line: Line):
    return CubicBezier(line.start, line.start + line.unit_tangent(), line.end - line.unit_tangent(), line.end)

def make_single_paths_cubic(single_paths: list):
    new_single_paths = single_paths.copy()
    for i, path in enumerate(new_single_paths):
        primitive = path[0]
        if not isinstance(primitive, CubicBezier):
            new_single_paths[i] = Path(line_to_cubic(primitive))

    return new_single_paths

def cubic_single_paths_to_relative_positions(cubic_single_paths, viewbox_scaling:float):
    all_timesteps = []
    for cubic_single_path in cubic_single_paths:
        start = cubic_single_path[0].start / viewbox_scaling
        c1 = cubic_single_path[0].control1 / viewbox_scaling
        c2 = cubic_single_path[0].control2 / viewbox_scaling
        end = cubic_single_path[0].end / viewbox_scaling
        timestep = torch.tensor([[start.real, start.imag], [c1.real, c1.imag], [c2.real, c2.imag], [end.real, end.imag]])
        all_timesteps.append(timestep)
    return torch.stack(all_timesteps)


data_path = "/scratch2/moritz_data/google_fonts_normalized/B"
all_svgs = glob(os.path.join(data_path, "*.svg"))
normalized_svg_path = random.choice(all_svgs)
print(normalized_svg_path)

paths, attributes, svg_attributes = svg2paths2(normalized_svg_path)
single_paths = get_single_paths(paths)

viewbox_scaling = float(svg_attributes["viewBox"].split(" ")[-2])
cubic_single_paths = make_single_paths_cubic(single_paths)
all_relative_positions = cubic_single_paths_to_relative_positions(cubic_single_paths, viewbox_scaling=viewbox_scaling).unsqueeze(0)
print(all_relative_positions.shape)

positions = all_relative_positions[:,:,[0,-1],:].flatten(start_dim=-2)  # (bs, t, 4, 2) -> (bs, t, 4)
generation_start_t = all_relative_positions.size(1) // 2
print(positions.shape)

input_bezier_points = all_relative_positions[:,:generation_start_t]
input_bezier_widths = torch.zeros(1,generation_start_t,1) + 2.0
max_new_steps = 20
scale = all_paths_to_max_diff([normalized_svg_path], index=0)
positions = positions

generation = model.generate_from_svg(input_bezier_points, input_bezier_widths, max_new_steps, scale, positions)

print("END")