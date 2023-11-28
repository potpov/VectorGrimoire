from matplotlib import pyplot as plt
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
from torchvision.utils import save_image, make_grid

print(torch.cuda.is_available())



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

def main(model:VectorGPTv2, normalized_svg_path, id:int=0, start_time: int = 0, out_dir:str = "images/vectorgtpv2"):
    paths, attributes, svg_attributes = svg2paths2(normalized_svg_path)
    single_paths = get_single_paths(paths)

    viewbox_scaling = float(svg_attributes["viewBox"].split(" ")[-2])
    cubic_single_paths = make_single_paths_cubic(single_paths)
    all_relative_positions = cubic_single_paths_to_relative_positions(cubic_single_paths, viewbox_scaling=viewbox_scaling).unsqueeze(0)
    print(all_relative_positions.shape)

    positions = all_relative_positions[:,:,[0,-1],:].flatten(start_dim=-2)  # (bs, t, 4, 2) -> (bs, t, 4)
    # generation_start_t = all_relative_positions.size(1) // 2
    print(positions.shape)

    input_bezier_points = all_relative_positions  # [:,:generation_start_t]
    input_bezier_widths = torch.zeros(1,input_bezier_points.size(1),1) + 2.0
    max_new_steps = 50
    scale = viewbox_scaling / all_paths_to_max_diff([normalized_svg_path], index=0)
    positions = positions

    all_final_preds = []

    for mode in ["auto_regressive", "teacher_forcing", "no_input"]:
        print("Generating for mode:", mode)
        out = model.generate_from_svg(input_bezier_points, input_bezier_widths, max_new_steps, scale, positions, mode=mode, start_time = start_time)

        generation = out[0]
        all_rasterized_shapes = out[1]
        final_gt_tensor = out[-1][0]

        generation_start_t = start_time + 1
        generation[0][generation_start_t] = torch.minimum(generation[0][generation_start_t], torch.zeros(*generation[0][generation_start_t].shape) + 0.8)

        save_image(make_grid(generation[0]), os.path.join(out_dir, f"{id}_{mode}_merged.png"))
        save_image(make_grid(all_rasterized_shapes[0]), os.path.join(out_dir, f"{id}_{mode}_centered_single.png"))

        all_final_preds.append({
            "mode": mode,
            "generation": generation[0][-1],  # only final timestep
        })
    save_comparison_plot(all_final_preds, final_gt_tensor, id)
    print("Finished with id:", id)
    return all_final_preds

def save_comparison_plot(all_final_preds_dicts, final_gt_tensor, id):
    """
    saves a plot of GT and each of the final predictions side-by-side
    """
    fig, axs = plt.subplots(1, len(all_final_preds_dicts)+1, figsize=(20,10))  # +1 for GT
    final_gt_image = final_gt_tensor.numpy().transpose(1,2,0)
    axs[0].imshow(final_gt_image, cmap="gray")
    axs[0].set_title("Ground Truth")
    for i, final_pred_dict in enumerate(all_final_preds_dicts):
        final_mse = torch.nn.functional.mse_loss(final_gt_tensor, final_pred_dict["generation"])
        img = final_pred_dict["generation"].numpy().transpose(1,2,0)
        axs[i+1].imshow(img)
        axs[i+1].set_title(final_pred_dict["mode"]+f", mse: {np.round(final_mse.item(), decimals=4)}")
    fig.savefig(f"images/vectorgtpv2/{id}_comparison.png")

if __name__ == '__main__':

    DATA_PATH = "/scratch2/moritz_data/google_fonts_normalized/B"
    NUM_SAMPLES = 5
    CKPT_PATH = "/scratch2/moritz_logs/VectorGPTv2/google_font/merged_input_pos_pred/checkpoints/last-v2.ckpt"
    SKIP_WEIGHT_LOADING = False
    CONFIG_PATH = "/home/mfeuerpfeil/master/thesis/configs/VectorGPTv2_overfit_B.yaml"
    BASE_OUT_DIR = "images/vectorgtpv2/v2_different_start_points"
    START_TIME = 0

    if not os.path.exists(BASE_OUT_DIR):
        os.makedirs(BASE_OUT_DIR)
    
    # remove all files in BASE_OUT_DIR after input from user
    input("Press Enter to continue deletion...")
    for file in os.listdir(BASE_OUT_DIR):
        os.remove(os.path.join(BASE_OUT_DIR, file))

    all_svgs = glob(os.path.join(DATA_PATH, "*.svg"))
    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    model = VectorGPTv2(**config['model_params'])

    if not SKIP_WEIGHT_LOADING:
        state_dict = torch.load(CKPT_PATH)["state_dict"]

        mapped_state_dict = {}
        for key in state_dict.keys():
            mapped_state_dict[key.replace("model.", "").replace("stop_predictor", "stop_predictor.model").replace("position_predictor", "position_predictor.model")] = state_dict[key]

        model.load_state_dict(mapped_state_dict, strict=True)
        print("[INFO] LOADED WEIGHTS for font checkpoint")
    model.eval()
    print()

    all_final_pred_dicts = []
    for i in range(NUM_SAMPLES):
        normalized_svg_path = random.choice(all_svgs)
        print(normalized_svg_path)
        final_pred_dicts = main(model, normalized_svg_path, id=i, start_time = START_TIME + 10 * i, out_dir = BASE_OUT_DIR)
        all_final_pred_dicts.extend(final_pred_dicts)

    auto_regressive_final_preds = [final_pred_dict["generation"] for final_pred_dict in all_final_pred_dicts if final_pred_dict["mode"] == "auto_regressive"]
    teacher_forcing_final_preds = [final_pred_dict["generation"] for final_pred_dict in all_final_pred_dicts if final_pred_dict["mode"] == "teacher_forcing"]
    no_input_final_preds = [final_pred_dict["generation"] for final_pred_dict in all_final_pred_dicts if final_pred_dict["mode"] == "no_input"]
    if len(auto_regressive_final_preds) > 0:
        save_image(make_grid(auto_regressive_final_preds), os.path.join(BASE_OUT_DIR, f"all_final_auto_regressive.png"))
    
    if len(teacher_forcing_final_preds) > 0:
        save_image(make_grid(teacher_forcing_final_preds), os.path.join(BASE_OUT_DIR, f"all_final_teacher_forcing.png"))
    
    if len(no_input_final_preds) > 0:
        save_image(make_grid(no_input_final_preds), os.path.join(BASE_OUT_DIR, f"all_final_no_input.png"))