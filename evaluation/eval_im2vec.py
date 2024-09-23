from typing import List
import torch
from torch import Tensor
from tqdm import tqdm
import yaml
from models.vaectorgen import VAEctorGen
from models.vector_vae_nlayers import VectorVAEnLayers
from dataset import MNISTDataset
from PIL import Image
import matplotlib.pyplot as plt
import os
import yaml
from dataset import VQDataModule, GenericRasterizedSVGDataset, GenericRasterDataset
from models import VQ_SVG_Stage2, VSQ
from tokenizer import VQTokenizer
from experiment import SVG_VQVAE_Stage2_Experiment
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch import Tensor
import pydiffvg
from torchvision.utils import make_grid, save_image
torch.cuda.is_available()
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor
from svg_fixing import get_fixed_svg_drawing, get_fixed_svg_render
import pandas as pd
from models import VectorVAEnLayers

import gc
import os
from typing import List
import yaml
from models import VQ_SVG_Stage2, VSQ
from tokenizer import VQTokenizer
from experiment import SVG_VQVAE_Stage2_Experiment
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from dataset import GenericRasterizedSVGDataset, VSQDatamodule, VQDataModule
from torch import nn
from math import ceil, sqrt
import time
import random
import argparse
from torchvision.utils import make_grid, save_image
torch.cuda.is_available()
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor
from svg_fixing import get_fixed_svg_drawing, get_fixed_svg_render, get_svg_render, min_dist_fix
import re

def map_wand_config(config):
    new_config = {}
    for k, v in config.items():
        if not "wandb" in k:
            new_config[k] = v["value"]
    return new_config

def save_im2vec_points_to_svg(model:VectorVAEnLayers,
                            all_points:List,
                            imsize,
                            save_base_dir,
                            filename):
        # z, log_var = model.encode(x)
        # all_points = model.decode(z)
        # print(all_points.std(dim=1))
        # all_points = ((all_points-0.5)*2 + 0.5)*self.imsize
        # if type(self.sort_idx) == type(None):
        #     angles = torch.atan(all_points[:,:,1]/all_points[:,:,0]).detach()
        #     self.sort_idx = torch.argsort(angles, dim=1)
        # Process the batch sequentially
        outputs = []
        shape_groups = []
        shapes = []
        for k in range(len(all_points)):
            # Get point parameters from network
            points = all_points[k].cpu()#[self.sort_idx[k]]
            if points.ndim > 2:
                points = points.squeeze(0)
            points = points * imsize
            color = torch.cat([torch.tensor([0,0,0,1]),])
            num_ctrl_pts = torch.zeros(model.curves, dtype=torch.int32) + 2

            path = pydiffvg.Path(
                num_control_points=num_ctrl_pts, points=points,
                is_closed=True)

            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(
                shape_ids=torch.tensor([len(shapes) - 1]),
                fill_color=None,
                stroke_color=color)
            shape_groups.append(path_group)
        pydiffvg.save_svg(f"{save_base_dir}/{filename}",
                            imsize, imsize, shapes, shape_groups)

class CLIPWrapper(nn.Module):
    def __init__(self, model, processor, device):
        super().__init__()
        self.device = device
        self.processor = processor
        self.model = model.to(self.device)

    @torch.no_grad()
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt")
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return self.model.get_image_features(**inputs)

@torch.no_grad()
def compute_fid_score(generated_images, real_images, device, model_str:str = "openai/clip-vit-base-patch32"):
    print(f"Computing FID with model {model_str} on device {device}")
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)
    wrapper = CLIPWrapper(model, processor, device)
    fid = FrechetInceptionDistance(feature=wrapper, normalize=True) # true is correct here
    fid = fid.to(device)
    bs = 32
    print("Adding generated images...")
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i+bs]).to(device)
        fid.update(generated_images_batch, real=False)
    print("Adding real images...")
    for i in tqdm(range(0, len(real_images), bs)):
        real_images_batch = torch.stack(real_images[i:i+bs]).to(device)
        fid.update(real_images_batch, real=True)

    return fid.compute()

@torch.no_grad()
def compute_clip_score(generated_images:List, captions:List, device, model_str:str = "openai/clip-vit-base-patch32"):
    print(f"Computing CLIP score with model {model_str} on device {device}")
    metric = CLIPScore(model_name_or_path=model_str)
    metric = metric.to(device)
    bs = 32
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i+bs]).to(device)
        captions_batch = captions[i:i+bs]
        metric.update(generated_images_batch, captions_batch)

    return metric.compute()

# # base_path = "/scratch2/moritz_logs/Im2Vec/figr8_star"
# # base_path = "/scratch2/moritz_logs/Im2Vec/fonts"
# base_path = "/scratch2/moritz_logs/Im2Vec/fonts_A_final"
# # base_path = "/scratch2/moritz_logs/Im2Vec/figr8"
# im2vec_model_path = os.path.join(base_path, "checkpoints/last.ckpt")
# out_base_dir = "/scratch2/moritz_logs/benchmark/im2vec/fonts_A_final"
# # out_base_dir = "/scratch2/moritz_logs/benchmark/im2vec/full_figr8"
# dataset = "fonts"
# class_name = "capital A"
fonts_a_config = {
    "base_path": "/scratch2/moritz_logs/Im2Vec/fonts_A_final",
    "im2vec_model_path": "checkpoints/last.ckpt",
    "out_base_dir": "/scratch2/moritz_logs/benchmark/im2vec/fonts_A_final",
    "dataset": "fonts",
    "class_name": "capital A"
}

fonts_full_config = {
    "base_path": "/scratch2/moritz_logs/Im2Vec/fonts",
    "im2vec_model_path": "checkpoints/last-v1.ckpt",
    "out_base_dir": "/scratch2/moritz_logs/benchmark/im2vec/full_fonts_final_final",
    "dataset": "fonts",
    "class_name": "glyph"
}

figr8_config = {
    "base_path": "/scratch2/moritz_logs/Im2Vec/figr8",
    "im2vec_model_path": "checkpoints/last-v2.ckpt",
    "out_base_dir": "/scratch2/moritz_logs/benchmark/im2vec/figr8",
    "dataset": "icons",
    "class_name": ""
}

im2vecsweep_base_config = {
    "base_path": "/scratch2/gesùbambino/im2vec",
    "im2vec_model_path": "last.ckpt",
    "im2vec_config_path": "config.yaml",
    "out_base_dir": "/scratch2/moritz_logs/benchmark/im2vec",
    "dataset": "icons",
    "class_name": "XXX"
}

mnist_5_config = {
    "base_path": "/raid/marco.cipriano/results/svg/im2vec/im2vec_mnist_num_5",
    "im2vec_model_path": "checkpoints/last.ckpt",
    "im2vec_config_path": "config.yaml",
    "out_base_dir": "/raid/marco.cipriano/results/svg/im2vec/im2vec_mnist_num_5/benchmark",
    "dataset": "mnist",
    "class_name": "5"
}

# ##>>>>>>>
selected_config = mnist_5_config
# ##<<<<<<<

# base_path = selected_config["base_path"]
# im2vec_model_path = os.path.join(base_path, selected_config["im2vec_model_path"])
# dataset = selected_config["dataset"]
# out_base_dir = selected_config["out_base_dir"]
# class_name = selected_config["class_name"]
# im2vec_config_path = os.path.join(base_path, "wandb/latest-run/files/config.yaml")


class_name = selected_config["class_name"]
print("doing now class_name: ", class_name)

base_path = os.path.join(selected_config["base_path"])
im2vec_model_path = os.path.join(base_path, selected_config["im2vec_model_path"])
im2vec_config_path = os.path.join(base_path, selected_config["im2vec_config_path"])
dataset = selected_config["dataset"]
out_base_dir = os.path.join(selected_config["out_base_dir"], class_name)


# class_name = "SVG"

assert dataset in ["fonts", "icons", "mnist"]
if dataset == "icons":
    get_prompt_template = lambda x: f"Black and white icon of {x}, vector graphic"
elif dataset == "mnist":
    get_prompt_template = lambda x: f"{str(x)} in black color"
else:
    get_prompt_template = lambda x: ""

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(im2vec_config_path, "r") as f:
    try:
        im2vec_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

# im2vec_config = map_wand_config(im2vec_config)

im2vec_config["model_params"]["imsize"] = im2vec_config["data_params"]["patch_size"]
im2vec_config["data_params"]["img_size"] = im2vec_config["data_params"]["patch_size"]

if dataset in ["fonts", "icons"]:
    ds = GenericRasterizedSVGDataset(**im2vec_config["data_params"], train=None)
else:
    data_module = MNISTDataset(**im2vec_config["data_params"], train=None)
    data_module.setup()
    ds = data_module.train_dataloader()

im2vec = VectorVAEnLayers(**im2vec_config["model_params"])
state_dict = torch.load(im2vec_model_path, map_location=device)["state_dict"]

num_samples = min(1000, len(ds))
# out_base_dir = "/scratch2/moritz_logs/benchmark/im2vec/fonts_a"
# out_base_dir = "/scratch2/moritz_logs/benchmark/im2vec/star_fig8_with_mse"
if os.path.exists(out_base_dir):
    # input(f"out_base_dir {out_base_dir} already exists, press enter to continue or CTRL+C to cancel")
    pass
for subdir in ["reconstructions", "samples", "gt"]:
    os.makedirs(os.path.join(out_base_dir, subdir), exist_ok=True)


try:
    im2vec.load_state_dict(state_dict)
except:
    im2vec.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
im2vec = im2vec.eval().to(device)
im2vec.base_control_features = im2vec.base_control_features.to(device)

original_images=[]
original_images_filled=[]
reconstruction_points=[]

IS_UNFILLED_AVAILABLE = dataset in ["fonts", "icons"]

# generate
with torch.no_grad():
    random.seed(42)
    random_idx = random.sample(range(len(ds)), num_samples)
    samples_points = im2vec.multishape_sample(num_samples, return_points=True, device=device)
    for i,idx in tqdm(enumerate(random_idx), total=len(random_idx)):
        gt_image = ds.dataset.__getitem__(idx)[0].to(device)
        reconstruction_points.append(im2vec.generate(gt_image, return_points=True))
        if dataset in ["mnist"]:
            IM_SIZE = im2vec_config["data_params"]["patch_size"]
            original_images_filled.append(gt_image.squeeze().cpu())
            save_image(gt_image, os.path.join(out_base_dir, "gt", f"gt_filled_{idx}.png"))
        elif dataset in ["fonts", "icons"]:
            IM_SIZE = 480
            filled_original = ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], IM_SIZE, fill=True)
            original_images_filled.append(filled_original)
            save_image(filled_original, os.path.join(out_base_dir,"gt",f"gt_filled_{idx}.png"))

            unfilled_original = ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], IM_SIZE, fill=False)
            original_images.append(unfilled_original)
            save_image(unfilled_original, os.path.join(out_base_dir,"gt",f"gt_unfilled_{idx}.png"))

        save_im2vec_points_to_svg(im2vec, samples_points[i], 72,os.path.join(out_base_dir,"samples"),f"im2vec_sample_{idx}.svg")
        save_im2vec_points_to_svg(im2vec, reconstruction_points[i], 72,os.path.join(out_base_dir,"reconstructions"),f"im2vec_reconstruction_{idx}.svg")

# evaluate
from utils import svg_file_path_to_tensor
svg_sample_paths = [os.path.join(out_base_dir,"samples",f"im2vec_sample_{idx}.svg") for idx in random_idx]
svg_reconstruction_paths = [os.path.join(out_base_dir,"reconstructions",f"im2vec_reconstruction_{idx}.svg") for idx in random_idx]

print("rendering svgs")
sample_renders_filled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=True) for p in svg_sample_paths]
reconstruction_renders_filled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=True) for p in svg_reconstruction_paths]

if IS_UNFILLED_AVAILABLE:
    sample_renders_unfilled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=False) for p in svg_sample_paths]
    reconstruction_renders_unfilled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=False) for p in svg_reconstruction_paths]

print("Computing MSE...")
mse_filled = torch.nn.functional.mse_loss(torch.stack(reconstruction_renders_filled), torch.stack(original_images_filled))
print(f"mse_filled: {mse_filled}")

if IS_UNFILLED_AVAILABLE:
    mse_unfilled = torch.nn.functional.mse_loss(torch.stack(reconstruction_renders_unfilled), torch.stack(original_images))
    print(f"mse_unfilled: {mse_unfilled}")


save_image(make_grid(original_images_filled, nrow=25), os.path.join(out_base_dir,"original_images_filled.png"))
save_image(make_grid(sample_renders_filled, nrow=25), os.path.join(out_base_dir,"sample_renders_filled.png"))
save_image(make_grid(reconstruction_renders_filled, nrow=25), os.path.join(out_base_dir,"reconstruction_renders_filled.png"))
if IS_UNFILLED_AVAILABLE:
    save_image(make_grid(original_images, nrow=25), os.path.join(out_base_dir,"original_images_unfilled.png"))
    save_image(make_grid(sample_renders_unfilled, nrow=25), os.path.join(out_base_dir,"sample_renders_unfilled.png"))
    save_image(make_grid(reconstruction_renders_unfilled, nrow=25), os.path.join(out_base_dir,"reconstruction_renders_unfilled.png"))

print("computing FID...")
fid_samples_filled = compute_fid_score(sample_renders_filled, original_images_filled, device)
fid_reconstructions_filled = compute_fid_score(reconstruction_renders_filled, original_images_filled, device)
print(f"fid_samples_filled: {fid_samples_filled}")
print(f"fid_reconstructions_filled: {fid_reconstructions_filled}")

if IS_UNFILLED_AVAILABLE:
    fid_samples_unfilled = compute_fid_score(sample_renders_unfilled, original_images, device)
    fid_reconstructions_unfilled = compute_fid_score(reconstruction_renders_unfilled, original_images, device)
    print(f"fid_samples_unfilled: {fid_samples_unfilled}")
    print(f"fid_reconstructions_unfilled: {fid_reconstructions_unfilled}")


print("computing CLIP score...")
if dataset in ["icons", "mnist"]:
    clip_samples_filled_prompt = compute_clip_score(sample_renders_filled, [get_prompt_template(class_name) for idx in random_idx], device)
    clip_reconstructions_filled_prompt = compute_clip_score(reconstruction_renders_filled, [get_prompt_template(class_name) for idx in random_idx], device)
    if IS_UNFILLED_AVAILABLE:
        clip_samples_unfilled_prompt = compute_clip_score(sample_renders_unfilled, [get_prompt_template(class_name) for idx in random_idx], device)
        clip_reconstructions_unfilled_prompt = compute_clip_score(reconstruction_renders_unfilled, [get_prompt_template(class_name) for idx in random_idx], device)
else:
    clip_samples_filled_prompt, clip_reconstructions_filled_prompt, clip_samples_unfilled_prompt, clip_reconstructions_unfilled_prompt = -1, -1, -1, -1

clip_samples_filled_class = compute_clip_score(sample_renders_filled, [class_name for idx in random_idx], device)
clip_reconstructions_filled_class = compute_clip_score(reconstruction_renders_filled, [class_name for idx in random_idx], device)
if IS_UNFILLED_AVAILABLE:
    clip_samples_unfilled_class = compute_clip_score(sample_renders_unfilled, [class_name for idx in random_idx], device)
    clip_reconstructions_unfilled_class = compute_clip_score(reconstruction_renders_unfilled, [class_name for idx in random_idx], device)

# clip_white_image_baseline = compute_clip_score([torch.ones(3,480,480) for idx in random_idx], ["star" for idx in random_idx], device)
# clip_black_image_baseline = compute_clip_score([torch.zeros(3,480,480) for idx in random_idx], ["star" for idx in random_idx], device)

clip_white_image_baseline, clip_black_image_baseline = -1, -1

print(f"clip_samples_filled_prompt: {clip_samples_filled_prompt}")
print(f"clip_reconstructions_filled_prompt: {clip_reconstructions_filled_prompt}")
if IS_UNFILLED_AVAILABLE:
    print(f"clip_samples_unfilled_prompt: {clip_samples_unfilled_prompt}")
    print(f"clip_reconstructions_unfilled_prompt: {clip_reconstructions_unfilled_prompt}")

print(f"clip_samples_filled_class: {clip_samples_filled_class}")
print(f"clip_reconstructions_filled_class: {clip_reconstructions_filled_class}")
if IS_UNFILLED_AVAILABLE:
    print(f"clip_samples_unfilled_class: {clip_samples_unfilled_class}")
    print(f"clip_reconstructions_unfilled_class: {clip_reconstructions_unfilled_class}")

print(f"clip_white_image_baseline: {clip_white_image_baseline}")
print(f"clip_black_image_baseline: {clip_black_image_baseline}")

with open(os.path.join(out_base_dir, "im2vec_results.txt"), "w") as f:
    f.write(f"num_samples: {len(random_idx)}\n")
    f.write(f"used dataset: {dataset}\n")
    f.write(f"class for clip: {class_name}\n")
    f.write(f"prompt template: {get_prompt_template('X')}\n\n")

    f.write(f"mse_recons_filled: \t{mse_filled}\n")
    f.write(f"fid_samples_filled: \t{fid_samples_filled}\n")
    f.write(f"fid_reconstructions_filled: \t{fid_reconstructions_filled}\n")
    f.write(f"clip_samples_filled_prompt: \t{clip_samples_filled_prompt}\n")
    f.write(f"clip_reconstructions_filled_prompt: \t{clip_reconstructions_filled_prompt}\n")
    f.write(f"clip_samples_filled_class: \t{clip_samples_filled_class}\n")
    f.write(f"clip_reconstructions_filled_class: \t{clip_reconstructions_filled_class}\n")
    f.write(f"clip_white_image_baseline: \t{clip_white_image_baseline}\n")
    f.write(f"clip_black_image_baseline: \t{clip_black_image_baseline}\n")
    if IS_UNFILLED_AVAILABLE:
        f.write(f"mse_recons_unfilled: \t{mse_unfilled}\n")
        f.write(f"fid_samples_unfilled: \t{fid_samples_unfilled}\n")
        f.write(f"fid_reconstructions_unfilled: \t{fid_reconstructions_unfilled}\n")
        f.write(f"clip_samples_unfilled_prompt: \t{clip_samples_unfilled_prompt}\n")
        f.write(f"clip_reconstructions_unfilled_prompt: \t{clip_reconstructions_unfilled_prompt}\n")
        f.write(f"clip_reconstructions_unfilled_class: \t{clip_reconstructions_unfilled_class}\n\n")
        f.write(f"clip_samples_unfilled_class: \t{clip_samples_unfilled_class}\n")

# also write config:
with open(os.path.join(out_base_dir, "im2vec_config.yaml"), "w") as f:
    yaml.dump(im2vec_config, f)
print("done")
