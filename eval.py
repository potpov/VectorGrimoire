import os
import yaml
from dataset import VQDataModule
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
from tqdm import tqdm

from torchvision.utils import make_grid, save_image
torch.cuda.is_available()
from utils import calculate_global_positions, shapes_to_drawing, drawing_to_tensor
from svg_fixing import get_fixed_svg_drawing, get_fixed_svg_render, get_svg_render

def map_wand_config(config):
    new_config = {}
    for k, v in config.items():
        if not "wandb" in k:
            new_config[k] = v["value"]
    return new_config

def load_stage2_model(config_path, ckpt_path, device,dataset:str = None, test_batch_size: int = 128):
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    if "wandb_version" in config.keys():
        config = map_wand_config(config)

    if dataset is not None:
        config["data_params"]["dataset"] = dataset

    vq_model = Vector_VQVAE(**config['stage1_params'], device = device)
    state_dict = torch.load(config['stage1_params']["checkpoint_path"])["state_dict"]
    try:
        vq_model.load_state_dict(state_dict)
    except:
        vq_model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    vq_model = vq_model.eval()
    tokenizer = VQTokenizer(vq_model, config["data_params"]["width"], 1, "bert-base-uncased", device = device)
    model = VQ_SVG_Stage2(tokenizer, **config['model_params'], device = device)
    state_dict = torch.load(ckpt_path)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        new_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        new_dict = {k.replace("transformer.", "transformer.model."): v for k, v in new_dict.items()}
        model.load_state_dict(new_dict)

    model = model.eval().to(device)
    text_only_tokenizer = VQTokenizer(None, config["data_params"]["width"], 1, "bert-base-uncased", use_text_encoder_only=True, codebook_size=tokenizer.codebook_size)
    data = VQDataModule(tokenizer = text_only_tokenizer, **config["data_params"], context_length=config['model_params']['max_seq_len'], test_batch_size = test_batch_size)
    data.setup(stage="test")
    return model, vq_model, tokenizer, data, config

def generate_test_set_stage2(model, tokenizer, ds, vq_context:int, temperature:float, device, n=None):
    model = model.eval()
    generated_images = []
    captions = []
    if n is None:
        n = len(ds)+1
    for text_tokens, attention_mask, vq_tokens, _, _ in tqdm(ds, total=n-1):
        bs = text_tokens.shape[0]
        text_tokens = text_tokens.to(device)
        attention_mask = attention_mask.to(device)
        if vq_context > 0:
            vq_tokens = vq_tokens[:,:vq_context].to(device)
        else:
            vq_tokens = torch.ones((bs, 1), device = device, dtype=torch.int64) * tokenizer.special_token_mapping.get("<BOS>")
        generation, reason = model.generate(text_tokens, attention_mask, vq_tokens, temperature = temperature)
        if generation.ndim > 1:
            generated_images.append([gen for gen in generation.cpu()])
            captions.append([tokenizer.decode_text(text_tok) for text_tok in text_tokens])
        else:
            generated_images.append(generation.cpu())
            captions.append(tokenizer.decode_text(text_tokens))
        if len(generated_images) >= n:
            break
    return generated_images, captions

def save_generations_with_captions(generations, captions, tokenizer, vq_context:int=0,title:str="", save_path = "generated_images.png"):
    ax_dim = int(np.ceil(np.sqrt(len(generations))))
    fig, axes = plt.subplots(ax_dim, ax_dim, figsize=(3*ax_dim, 3*ax_dim))
    for i, ax in enumerate(axes.flatten()):
        bezier_points, positions = tokenizer.decode(generations[i].to(tokenizer.device), ignore_special_tokens=False)
        ax.imshow(get_svg_render(bezier_points, positions, num_strokes_to_paint=vq_context).permute(1, 2, 0))
        ax.set_title(captions[i])
        ax.axis('off')

    fig.suptitle(title)
    fig.savefig(save_path, dpi=300)

from math import ceil, sqrt
import random


def save_svg(tokenizer:VQTokenizer, 
             bezier_points: Tensor, 
             center_positions: Tensor, 
             padded_individual_max_length: float, 
             stroke_width: float, 
             save_path:str,
             w: float = 128, 
             num_strokes_to_paint: int = 0,
             fixing_method:str=None,):
    assert fixing_method in [None, "min_dist_clip", "min_dist_interpolate"], "fixing_method must be one of None, 'min_dist_clip', 'min_dist_interpolate'"
    if fixing_method is None:
        drawing = tokenizer.assemble_svg(bezier_points, center_positions, padded_individual_max_length, stroke_width, w, num_strokes_to_paint)
    else:
        drawing = get_fixed_svg_drawing(bezier_points, 
                                        center_positions,
                                        method=fixing_method, 
                                        padded_individual_max_length=padded_individual_max_length, 
                                        stroke_width=stroke_width, 
                                        width=w, 
                                        num_strokes_to_paint=num_strokes_to_paint)
    drawing.saveas(save_path, pretty=True) 

from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import AutoProcessor, CLIPModel
from dataset import GenericRasterizedSVGDataset
from torch import nn
class CLIPWrapper(nn.Module):
    def __init__(self, model, processor, device):
        super().__init__()
        self.device = device
        self.processor = processor
        self.model = model.to(self.device)

    @torch.no_grad()
    def forward(self, x):
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.device)
        return self.model.get_image_features(**inputs)

@torch.no_grad()
def compute_fid_score(generated_images, real_images, device, model_str:str = "openai/clip-vit-base-patch32"):
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)
    wrapper = CLIPWrapper(model, processor, device)
    fid = FrechetInceptionDistance(feature=wrapper, normalize=True)
    fid = fid.to(device)
    generated_images = torch.stack(generated_images).to(device)
    real_images = torch.stack(real_images).to(device)
    fid.update(generated_images, real=False)
    fid.update(real_images, real=True)
    return fid.compute()

def benchmark_stage2_sgamo(config_path:str, 
                           ckpt_path:str, 
                           dataset:str, 
                           out_dir:str, 
                           vq_context:int, 
                           padded_individual_max_length:float, 
                           stroke_width:float,
                           num_batches:int,
                           test_batch_size:int, 
                           max_num_svgs:int, 
                           device, 
                           temperature:float = 0.0,):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print("Loading stage2 model...")
    model, vq_model, tokenizer, data, config = load_stage2_model(config_path, ckpt_path, device, dataset=dataset, test_batch_size=test_batch_size)

    print("Generating test set...")
    generated_vq_tokens, prompts = generate_test_set_stage2(model, tokenizer, data.test_dataloader(), vq_context = vq_context, temperature = temperature, device = device, n=num_batches)

    flattened_generated_vq_tokens = [gen for sublist in generated_vq_tokens for gen in sublist]
    flattened_prompts = [cap for sublist in prompts for cap in sublist]

    print("Rasterizing stage 2 generations...")
    # each svg has bezier_points and positions
    generated_svgs = [tokenizer.decode(x.to(tokenizer.device), ignore_special_tokens=False) for x in flattened_generated_vq_tokens]
    unfixed_renderings = [get_svg_render(bezier_points, positions, num_strokes_to_paint=vq_context) for bezier_points, positions in generated_svgs]
    pc_fixed_renderings = [get_fixed_svg_render(bezier_points, positions, num_strokes_to_paint=vq_context, method="min_dist_clip") for bezier_points, positions in generated_svgs]
    pi_fixed_renderings = [get_fixed_svg_render(bezier_points, positions, num_strokes_to_paint=vq_context, method="min_dist_interpolate") for bezier_points, positions in generated_svgs]
    
    rasterized_ds = GenericRasterizedSVGDataset(config["data_params"]["csv_path"],
                                    train=None,
                                    fill=False,
                                    img_size=480)

    print("Computing FID score...")
    unfixed_fid_score = compute_fid_score(unfixed_renderings, [rasterized_ds[i][0] for i in range(len(rasterized_ds))], device)
    pc_fixed_fid_score = compute_fid_score(pc_fixed_renderings, [rasterized_ds[i][0] for i in range(len(rasterized_ds))], device)
    pi_fixed_fid_score = compute_fid_score(pi_fixed_renderings, [rasterized_ds[i][0] for i in range(len(rasterized_ds))], device)

    print(f"Unfixed FID: {unfixed_fid_score}")
    print(f"PC fixed FID: {pc_fixed_fid_score}")
    print(f"PI fixed FID: {pi_fixed_fid_score}")

    with open(os.path.join(out_dir, "results_fid_sgamo.txt"), "w+") as f:
        f.write(f"num_samples: {num_batches*test_batch_size}\n")
        f.write(f"Unfixed FID: {unfixed_fid_score}\n")
        f.write(f"PC fixed FID: {pc_fixed_fid_score}\n")
        f.write(f"PI fixed FID: {pi_fixed_fid_score}\n")


    print("Saving stage 2 generations...")
    os.makedirs(os.path.join(out_dir,"svgs","unfixed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"svgs","pc_fixed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"svgs","pi_fixed"), exist_ok=True)
    prompt_string = "\n".join(flattened_prompts)
    with open(os.path.join(out_dir,"svgs","prompts.txt"), "w") as f:
        f.write(prompt_string)
    for i, (bezier_points, positions) in enumerate(generated_svgs):
        if i >= max_num_svgs:
            break
        save_svg(tokenizer, bezier_points, positions, padded_individual_max_length, stroke_width, os.path.join(out_dir,"svgs","unfixed", f"unfixed_{i}.svg"), num_strokes_to_paint=vq_context)
        save_svg(tokenizer, bezier_points, positions, padded_individual_max_length, stroke_width, os.path.join(out_dir,"svgs","pc_fixed", f"pc_fixed_{i}.svg"), num_strokes_to_paint=vq_context, fixing_method="min_dist_clip")
        save_svg(tokenizer, bezier_points, positions, padded_individual_max_length, stroke_width, os.path.join(out_dir,"svgs", "pi_fixed",f"pi_fixed_{i}.svg"), num_strokes_to_paint=vq_context, fixing_method="min_dist_interpolate")

    max_single_image = min(100, len(unfixed_renderings))
    save_image(make_grid(unfixed_renderings[:max_single_image], nrow=int(ceil(sqrt(max_single_image)))), os.path.join(out_dir,"unfixed_renderings.png"))
    save_image(make_grid(pc_fixed_renderings[:max_single_image], nrow=int(ceil(sqrt(max_single_image)))), os.path.join(out_dir,"pc_fixed_renderings.png"))
    save_image(make_grid(pi_fixed_renderings[:max_single_image], nrow=int(ceil(sqrt(max_single_image)))), os.path.join(out_dir,"pi_fixed_renderings.png"))

def main():
    stage2_config_path = "/scratch2/moritz_logs/SVG_VQVAE/Stage2/filtered_fonts_full_single_code/wandb/run-20240226_191349-ro48a2jp/files/config.yaml"
    stage2_ckpt_path = "/scratch2/moritz_logs/SVG_VQVAE/Stage2/filtered_fonts_full_single_code/checkpoints/last-v1.ckpt"
    out_dir = "images/benchmark/sgamo/stage2/fonts"
    vq_context = 0
    padded_individual_max_length = 9.5
    stroke_width = 0.4
    num_batches = 100
    test_batch_size = 64
    max_num_svgs = 2000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Will be generating {num_batches*test_batch_size} samples...")

    benchmark_stage2_sgamo(stage2_config_path,
                           stage2_ckpt_path, 
                           "fonts", 
                           out_dir, 
                           vq_context, 
                           padded_individual_max_length, 
                           stroke_width, 
                           num_batches, 
                           test_batch_size, 
                           max_num_svgs, 
                           device,
                           temperature=0.0)

if __name__ == "__main__":
    main()