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
from models import VQ_SVG_Stage2, Vector_VQVAE
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


base_path = "/scratch2/moritz_logs/Im2Vec/figr8_star"
im2vec_config_path = os.path.join(base_path, "wandb/latest-run/files/config.yaml")
im2vec_model_path = os.path.join(base_path, "checkpoints/last.ckpt")
device = "cuda" if torch.cuda.is_available() else "cpu"

with open(im2vec_config_path, "r") as f:
    try:
        im2vec_config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(exc)

im2vec_config = map_wand_config(im2vec_config)

ds = GenericRasterizedSVGDataset(**im2vec_config["data_params"], train=None)
im2vec = VectorVAEnLayers(**im2vec_config["model_params"])
state_dict = torch.load(im2vec_model_path)["state_dict"]

num_samples = min(10, len(ds))

try:
    im2vec.load_state_dict(state_dict)
except:
    im2vec.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
im2vec = im2vec.eval().to(device)
im2vec.base_control_features = im2vec.base_control_features.to(device)





original_images=[]
reconstruction_points=[]

# out_base_dir = "/scratch2/moritz_logs/benchmark/im2vec/star_fig8"
out_base_dir = "/home/mfeuerpfeil/master/thesis/images/im2vec/star_fig8"
for subdir in ["reconstructions", "samples", "originals"]:
    os.makedirs(os.path.join(out_base_dir, subdir), exist_ok=True)


with torch.no_grad():
    random.seed(42)
    random_idx = random.sample(range(len(ds)), num_samples)
    samples_points = im2vec.multishape_sample(num_samples, return_points=True, device=device)
    for i,idx in tqdm(enumerate(random_idx), total=len(random_idx)):
        gt_image = ds[idx][0].to(device)
        reconstruction_points.append(im2vec.generate(gt_image.unsqueeze(0), return_points=True))
        save_image(ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], 480, fill=True), os.path.join(out_base_dir,"originals",f"original_{idx}_filled.png"))
        save_image(ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], 480, fill=False), os.path.join(out_base_dir,"originals",f"original_{idx}.png"))
        save_im2vec_points_to_svg(im2vec, samples_points[i], 72,os.path.join(out_base_dir,"samples"),f"im2vec_sample_{idx}.svg")
        save_im2vec_points_to_svg(im2vec, reconstruction_points[i], 72,os.path.join(out_base_dir,"reconstructions"),f"im2vec_reconstruction_{idx}.svg")
print("done")