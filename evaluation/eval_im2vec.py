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

import pydiffvg

from models import VectorVAEnLayers

import gc
import os
from typing import List
import yaml
import torch
from tqdm import tqdm
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


#############
# CONFIGS
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
    "class_name": "5",  # set this to None if you wanna use the full dataset: None, 5
}

mnist_0_config = {
    "base_path": "/raid/marco.cipriano/results/svg/im2vec/im2vec_mnist_zero_T10",
    "im2vec_model_path": "checkpoints/last.ckpt",
    "im2vec_config_path": "config.yaml",
    "out_base_dir": "/raid/marco.cipriano/results/svg/im2vec/im2vec_mnist_zero_T10/benchmark",
    "dataset": "mnist",
    "class_name": "0",  # set this to None if you wanna use the full dataset: None, 5
}

mnist_full_config = {
    "base_path": "/raid/marco.cipriano/results/svg/im2vec/im2vec_mnist_full_T10",
    "im2vec_model_path": "checkpoints/last.ckpt",
    "im2vec_config_path": "config.yaml",
    "out_base_dir": "/raid/marco.cipriano/results/svg/im2vec/im2vec_mnist_full_T10/benchmark",
    "dataset": "mnist",
    "class_name": None
}

config_dict = {
    'mnist_full_config': mnist_full_config,
    'mnist_0_config': mnist_0_config,
    'mnist_5_config': mnist_5_config,
    'im2vecsweep_base_config': im2vecsweep_base_config,
    'figr8_config': figr8_config,
    'fonts_full_config': fonts_full_config,
    'fonts_a_config': fonts_a_config,
}


def evaluate(conf_key):

    # ##>>>>>>>
    print("USING CONFIG: ", conf_key)
    selected_config = config_dict[conf_key]
    # ##<<<<<<<

    ### GLOBAL CONF FOR CONSISTENCY WITH OUR BENCHMARK
    IM_SIZE = 128  # mnist -> 128, other datasets were tested with 480
    EVALUATION_SAMPLES = 5000

    if selected_config["class_name"] is None:
        print("LOADING THE FULL DATASET!")
        class_name = "full"
        subset = None
    else:
        subset = selected_config["class_name"]
        class_name = selected_config["class_name"]

    print("doing now class_name: ", class_name)
    base_path = os.path.join(selected_config["base_path"])
    im2vec_model_path = os.path.join(base_path, selected_config["im2vec_model_path"])
    im2vec_config_path = os.path.join(base_path, selected_config["im2vec_config_path"])
    dataset = selected_config["dataset"]
    out_base_dir = os.path.join(selected_config["out_base_dir"], class_name)


    assert dataset in ["fonts", "icons", "mnist"]
    if dataset == "icons":
        get_prompt_template = lambda x: f"Black and white icon of {x}, vector graphic"
    elif dataset == "mnist":
        get_prompt_template = lambda x: f"{str(x)}"
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
        im2vec_config["data_params"]["patch_size"] = IM_SIZE  # forcing image size to our config
        im2vec_config["data_params"]["subset"] = subset  # forcing image size to our config
        data_module = MNISTDataset(**im2vec_config["data_params"], train=None)

        data_module.setup()
        ds = data_module.test_dataloader()

    im2vec = VectorVAEnLayers(**im2vec_config["model_params"])
    state_dict = torch.load(im2vec_model_path, map_location=device)["state_dict"]

    num_samples = min(EVALUATION_SAMPLES, len(ds.dataset.labels))

    if os.path.exists(out_base_dir):
        print("WARNING: PATH ALREADY EXISTS, OVERWRITING!")

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
    captions = []
    # generate
    with torch.no_grad():
        random.seed(42)
        random_idx = random.sample(range(len(ds.dataset.labels)), num_samples)
        samples_points = im2vec.multishape_sample(num_samples, return_points=True, device=device)
        for i,idx in tqdm(enumerate(random_idx), total=len(random_idx)):
            batch = ds.dataset.__getitem__(idx)
            imgs, label, _, description = batch
            captions.append(f"{str(label)} in black color")
            gt_image = imgs.to(device)
            reconstruction_points.append(im2vec.generate(gt_image, return_points=True))
            if dataset in ["mnist"]:
                original_images_filled.append(gt_image.squeeze().cpu())
                save_image(gt_image, os.path.join(out_base_dir, "gt", f"gt_filled_{idx}.png"))
            elif dataset in ["fonts", "icons"]:
                filled_original = ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], IM_SIZE, fill=True)
                original_images_filled.append(filled_original)
                save_image(filled_original, os.path.join(out_base_dir,"gt",f"gt_filled_{idx}.png"))

                unfilled_original = ds._rasterize_svg(ds.df.iloc[idx]["simplified_svg_file_path"], IM_SIZE, fill=False)
                original_images.append(unfilled_original)
                save_image(unfilled_original, os.path.join(out_base_dir,"gt",f"gt_unfilled_{idx}.png"))

            save_im2vec_points_to_svg(im2vec, samples_points[i], 72, os.path.join(out_base_dir, "samples"), f"im2vec_sample_{idx}.svg")
            save_im2vec_points_to_svg(im2vec, reconstruction_points[i], 72, os.path.join(out_base_dir, "reconstructions"), f"im2vec_reconstruction_{idx}.svg")

    # evaluate
    from utils import svg_file_path_to_tensor
    svg_sample_paths = [os.path.join(out_base_dir,"samples",f"im2vec_sample_{idx}.svg") for idx in random_idx]
    svg_reconstruction_paths = [os.path.join(out_base_dir,"reconstructions",f"im2vec_reconstruction_{idx}.svg") for idx in random_idx]

    print("rendering svgs")
    sample_renders_filled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=True) for p in svg_sample_paths]
    reconstruction_renders_filled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=True) for p in svg_reconstruction_paths]

    sample_renders_unfilled = [svg_file_path_to_tensor(p, stroke_width=0.4, image_size=IM_SIZE, filling=False) for p in svg_sample_paths]
    if IS_UNFILLED_AVAILABLE:
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
    save_image(make_grid(sample_renders_unfilled, nrow=25), os.path.join(out_base_dir,"sample_renders_unfilled.png"))
    if IS_UNFILLED_AVAILABLE:
        save_image(make_grid(original_images, nrow=25), os.path.join(out_base_dir,"original_images_unfilled.png"))
        save_image(make_grid(reconstruction_renders_unfilled, nrow=25), os.path.join(out_base_dir,"reconstruction_renders_unfilled.png"))

    print("computing FID...")
    fid_samples_filled = compute_fid_score(sample_renders_filled, original_images_filled, device)
    fid_reconstructions_filled = compute_fid_score(reconstruction_renders_filled, original_images_filled, device)
    print(f"fid_samples_filled: {fid_samples_filled}")
    print(f"fid_reconstructions_filled: {fid_reconstructions_filled}")

    if IS_UNFILLED_AVAILABLE:
        fid_samples_unfilled = compute_fid_score(sample_renders_unfilled, original_images, device)
        print(f"fid_samples_unfilled: {fid_samples_unfilled}")
        fid_reconstructions_unfilled = compute_fid_score(reconstruction_renders_unfilled, original_images, device)
        print(f"fid_reconstructions_unfilled: {fid_reconstructions_unfilled}")


    print("computing CLIP score...")
    if dataset in ["icons", "mnist"]:
        clip_samples_filled_prompt = compute_clip_score(sample_renders_filled, [get_prompt_template(class_name) for idx in random_idx], device)
        clip_reconstructions_filled_prompt = compute_clip_score(reconstruction_renders_filled, [get_prompt_template(class_name) for idx in random_idx], device)
        if IS_UNFILLED_AVAILABLE:
            clip_samples_unfilled_prompt = compute_clip_score(sample_renders_unfilled, [get_prompt_template(class_name) for idx in random_idx], device)
            clip_reconstructions_unfilled_prompt = compute_clip_score(reconstruction_renders_unfilled, [get_prompt_template(class_name) for idx in random_idx], device)
    else:
        clip_samples_filled_prompt = compute_clip_score(sample_renders_filled, captions, device)

    clip_samples_unfilled_prompt = compute_clip_score(sample_renders_unfilled, captions, device)

    print(f"CLIP filled (GEN): {clip_samples_filled_prompt}")
    print(f"CLIP filled (RECON): {clip_reconstructions_filled_prompt}")
    print(f"clip unfilled (GEN): {clip_samples_unfilled_prompt}")
    if IS_UNFILLED_AVAILABLE:
        print(f"CLIP unfilled (RECON): {clip_reconstructions_unfilled_prompt}")


    with open(os.path.join(out_base_dir, "im2vec_results.txt"), "w") as f:
        f.write(f"num_samples: {len(random_idx)}\n")
        f.write(f"used dataset: {dataset}\n")
        f.write(f"subset for clip: {class_name}\n")

        f.write(f"mse_recons_filled: \t{mse_filled}\n")
        f.write(f"fid_samples_filled: \t{fid_samples_filled}\n")
        f.write(f"fid_reconstructions_filled: \t{fid_reconstructions_filled}\n")
        f.write(f"clip_samples_filled: \t{clip_samples_filled_prompt}\n")
        f.write(f"clip_reconstructions_filled: \t{clip_reconstructions_filled_prompt}\n")
        f.write(f"clip_samples_unfilled_prompt: \t{clip_samples_unfilled_prompt}\n")
        if IS_UNFILLED_AVAILABLE:
            f.write(f"fid_samples_unfilled: \t{fid_samples_unfilled}\n")
            f.write(f"mse_recons_unfilled: \t{mse_unfilled}\n")
            f.write(f"fid_reconstructions_unfilled: \t{fid_reconstructions_unfilled}\n")
            f.write(f"clip_reconstructions_unfilled_prompt: \t{clip_reconstructions_unfilled_prompt}\n")

    # also write config:
    with open(os.path.join(out_base_dir, "im2vec_config.yaml"), "w") as f:
        yaml.dump(im2vec_config, f)
    print("done, results saved to ", out_base_dir)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Get configuration by key.')
    parser.add_argument('--configuration', type=str, help='Configuration key to use')
    # Parse arguments
    args = parser.parse_args()

    # Retrieve the configuration object using the provided key
    config_key = args.configuration
    assert config_key in config_dict, f"Configuration key {config_key} not found"
    # 'mnist_full_config', 'mnist_0_config','mnist_5_config',
    # 'im2vecsweep_base_config','figr8_config', 'fonts_full_config','fonts_a_config',
    evaluate(config_key)