import gc
import os
from typing import List
import yaml
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
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from dataset import GenericRasterizedSVGDataset, GlyphazznStage1Datamodule, VQDataModule
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
    # model.load_state_dict(state_dict)
    try:
        model.load_state_dict(state_dict)
    except:
        new_dict = state_dict
        # new_dict = {k.replace("transformer.model.", "model.transformer.model."): v for k, v in new_dict.items()}
        # new_dict = {k.replace("text_embedder.", "model.text_embedder."): v for k, v in new_dict.items()}
        new_dict = {k.replace("model.transformer.model.","transformer.model."): v for k, v in new_dict.items()}
        new_dict = {k.replace("model.text_embedder.","text_embedder."): v for k, v in new_dict.items()}
        new_dict = {k.replace("model.pos_emb.","pos_emb."): v for k, v in new_dict.items()}
        new_dict = {k.replace("model.vq_embedding.","vq_embedding."): v for k, v in new_dict.items()}
        new_dict = {k.replace("model.mapping_layer.","mapping_layer."): v for k, v in new_dict.items()}
        new_dict = {k.replace("model.final_linear.","final_linear."): v for k, v in new_dict.items()}
        # new_dict = {k.replace("tokenizer.vq_model", "tokenizer.vq_model.encoder"): v for k, v in new_dict.items()}
        # new_dict = {k.replace("tokenizer.vq_decoder", "tokenizer.vq_model.decoder"): v for k, v in new_dict.items()}
        # new_dict = {k.replace("model.", "",1): v for k, v in state_dict.items()}
        pattern = r"(\.ff\.2\.)(weight|bias)"
        replacement = r".ff.3.\2"
        new_dict = {re.sub(pattern,replacement,k): v for k, v in new_dict.items()}

        model.load_state_dict(new_dict)

    model = model.eval().to(device)
    text_only_tokenizer = VQTokenizer(None, config["data_params"]["width"], 1, "bert-base-uncased", use_text_encoder_only=True, codebook_size=tokenizer.codebook_size)
    data = VQDataModule(tokenizer = text_only_tokenizer, **config["data_params"], context_length=config['model_params']['max_seq_len'], test_batch_size = test_batch_size)
    data.setup(stage="test")
    return model, vq_model, tokenizer, data, config

def generate_test_set_stage2(model, tokenizer, dl:DataLoader, vq_context:int, temperature:float, device, n=None):
    model = model.eval()
    generated_shapes = []
    captions = []
    if n is None:
        n = len(dl)+1
    for text_tokens, attention_mask, vq_tokens, _, _ in tqdm(dl, total=n-1):
        bs = text_tokens.shape[0]
        text_tokens = text_tokens.to(device)
        attention_mask = attention_mask.to(device)
        if vq_context > 0:
            vq_tokens = vq_tokens[:,:vq_context].to(device)
        else:
            vq_tokens = torch.ones((bs, 1), device = device, dtype=torch.int64) * tokenizer.special_token_mapping.get("<BOS>")
        generation, reason = model.generate(text_tokens, attention_mask, vq_tokens, temperature = temperature)
        if generation.ndim > 1:
            generated_shapes.append([gen for gen in generation.cpu()])
            captions.append([tokenizer.decode_text(text_tok) for text_tok in text_tokens])
        else:
            generated_shapes.append(generation.cpu())
            captions.append(tokenizer.decode_text(text_tokens))
        if len(generated_shapes) >= n:
            break
    return generated_shapes, captions

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
    print(f"Computing FID with model {model_str} on device {device}")
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)
    wrapper = CLIPWrapper(model, processor, device)
    fid = FrechetInceptionDistance(feature=wrapper, normalize=True)
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

def benchmark_stage2_sgamo(config_path:str, 
                           ckpt_path:str, 
                           dataset:str, 
                           out_dir:str, 
                           vq_context:int, 
                           padded_individual_max_length:float, 
                           stroke_width:float,
                           num_batches:int,
                           num_real_images:int,
                           test_batch_size:int, 
                           max_num_svgs:int, 
                           device, 
                           temperature:float = 0.0,
                           clip_model = "openai/clip-vit-base-patch32",
                           **kwargs):
    print("received unused arguments: ", kwargs)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"Warning: {out_dir} already exists. Overwriting files...")

    print("Loading stage2 model...")
    model, vq_model, tokenizer, data, config = load_stage2_model(config_path, ckpt_path, device, dataset=dataset, test_batch_size=test_batch_size)

    print("Generating test set...")
    generated_vq_tokens, prompts = generate_test_set_stage2(model, tokenizer, data.test_dataloader(), vq_context = vq_context, temperature = temperature, device = device, n=num_batches)

    for prompt in prompts:
        assert len(prompt) > 1, f"Prompts must not be empty, got: {prompt}"

    flattened_generated_vq_tokens = [gen for sublist in generated_vq_tokens for gen in sublist]
    flattened_prompts = [cap for sublist in prompts for cap in sublist]

    print("Decoding tokens into shapes...")
    generated_svgs = []
    for x in tqdm(flattened_generated_vq_tokens):
        generated_svgs.append(tokenizer.decode(x.to(tokenizer.device), ignore_special_tokens=False))
    
    print("Fixing svgs...")
    unfixed_renderings = []
    pc_fixed_renderings = []
    pi_fixed_renderings = []
    for bezier_points, positions in tqdm(generated_svgs):
        num_strokes_to_paint = min(vq_context, len(positions))
        unfixed_renderings.append(get_svg_render(bezier_points, positions, num_strokes_to_paint=num_strokes_to_paint))
        pc_fixed_renderings.append(get_fixed_svg_render(bezier_points, positions, num_strokes_to_paint=num_strokes_to_paint, method="min_dist_clip"))
        pi_fixed_renderings.append(get_fixed_svg_render(bezier_points, positions, num_strokes_to_paint=num_strokes_to_paint, method="min_dist_interpolate"))
    
    rasterized_ds = GenericRasterizedSVGDataset(config["data_params"]["csv_path"],
                                    train=None,
                                    fill=False,
                                    img_size=480,
                                    stroke_width=stroke_width,)
    
    print("Loading rasterized GT images...")
    random.seed(42)
    indices = random.sample(range(len(rasterized_ds)), num_real_images)
    real_imgs = []
    for i in tqdm(indices):
        real_imgs.append(rasterized_ds[i][0])

    print("Computing FID score...")
    unfixed_fid_score = compute_fid_score(unfixed_renderings, real_imgs, device, model_str = clip_model)
    pc_fixed_fid_score = compute_fid_score(pc_fixed_renderings, real_imgs, device, model_str = clip_model)
    pi_fixed_fid_score = compute_fid_score(pi_fixed_renderings, real_imgs, device, model_str = clip_model)

    real_imgs_grid = make_grid(real_imgs[:10],nrow=10)
    unfixed_grid = make_grid(unfixed_renderings[:10],nrow=10)
    pc_fixed_grid = make_grid(pc_fixed_renderings[:10],nrow=10)
    pi_fixed_grid = make_grid(pi_fixed_renderings[:10],nrow=10)
    [save_image(x, os.path.join(out_dir, f"{name}.png")) for x, name in zip([real_imgs_grid, unfixed_grid, pc_fixed_grid, pi_fixed_grid], ["real_imgs", "unfixed", "pc_fixed", "pi_fixed"])]


    # white_baseline_fid_score = compute_fid_score([torch.ones((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(real_imgs))], real_imgs, device, model_str = clip_model)
    # black_baseline_fid_score = compute_fid_score([torch.zeros((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(real_imgs))], real_imgs, device, model_str = clip_model)

    print(f"Unfixed FID: {unfixed_fid_score}")
    print(f"PC fixed FID: {pc_fixed_fid_score}")
    print(f"PI fixed FID: {pi_fixed_fid_score}")
    # print(f"White baseline FID: {white_baseline_fid_score}")
    # print(f"Black baseline FID: {black_baseline_fid_score}")

    with open(os.path.join(out_dir, "results_fid_sgamo.txt"), "w+") as f:
        f.write(f"num_samples: {num_batches*test_batch_size}\n")
        f.write(f"Unfixed FID: {unfixed_fid_score}\n")
        f.write(f"PC fixed FID: {pc_fixed_fid_score}\n")
        f.write(f"PI fixed FID: {pi_fixed_fid_score}\n")
        # f.write(f"White baseline FID: {white_baseline_fid_score}\n")
        # f.write(f"Black baseline FID: {black_baseline_fid_score}\n")

    print("Computing CLIP scores...")
    get_prompt_template = lambda x: f"Black and white icon of {x}, vector art"
    clip_adjusted_prompts = [get_prompt_template(x) for x in flattened_prompts]
    unfixed_clip_score = compute_clip_score(unfixed_renderings, flattened_prompts, device, model_str = clip_model)
    pc_fixed_clip_score = compute_clip_score(pc_fixed_renderings, flattened_prompts, device, model_str = clip_model)
    pi_fixed_clip_score = compute_clip_score(pi_fixed_renderings, flattened_prompts, device, model_str = clip_model)
    prompt_adjusted_unfixed_clip_score = compute_clip_score(unfixed_renderings, clip_adjusted_prompts, device, model_str = clip_model)
    prompt_adjusted_pc_fixed_clip_score = compute_clip_score(pc_fixed_renderings, clip_adjusted_prompts, device, model_str = clip_model)
    prompt_adjusted_pi_fixed_clip_score = compute_clip_score(pi_fixed_renderings, clip_adjusted_prompts, device, model_str = clip_model)

    # white_baseline_clip_score = compute_clip_score([torch.ones((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(flattened_prompts))], flattened_prompts, device, model_str = clip_model)
    # black_baseline_clip_score = compute_clip_score([torch.zeros((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(flattened_prompts))], flattened_prompts, device, model_str = clip_model)

    print(f"Unfixed CLIP score: {unfixed_clip_score}")
    print(f"PC fixed CLIP score: {pc_fixed_clip_score}")
    print(f"PI fixed CLIP score: {pi_fixed_clip_score}")
    print(f"Prompt adjusted unfixed CLIP score: {prompt_adjusted_unfixed_clip_score}")
    print(f"Prompt adjusted PC fixed CLIP score: {prompt_adjusted_pc_fixed_clip_score}")
    print(f"Prompt adjusted PI fixed CLIP score: {prompt_adjusted_pi_fixed_clip_score}")
    # print(f"White baseline CLIP score: {white_baseline_clip_score}")
    # print(f"Black baseline CLIP score: {black_baseline_clip_score}")

    with open(os.path.join(out_dir, "results_clip_sgamo.txt"), "w+") as f:
        f.write(f"num_samples: {num_batches*test_batch_size}\n")
        f.write("Prompt adjusted template: "+get_prompt_template("X")+"\n")
        f.write(f"Unfixed CLIP score: {unfixed_clip_score}\n")
        f.write(f"PC fixed CLIP score: {pc_fixed_clip_score}\n")
        f.write(f"PI fixed CLIP score: {pi_fixed_clip_score}\n")
        f.write(f"Prompt adjusted unfixed CLIP score: {prompt_adjusted_unfixed_clip_score}\n")
        f.write(f"Prompt adjusted PC fixed CLIP score: {prompt_adjusted_pc_fixed_clip_score}\n")
        f.write(f"Prompt adjusted PI fixed CLIP score: {prompt_adjusted_pi_fixed_clip_score}\n")
        # f.write(f"White baseline CLIP score: {white_baseline_clip_score}\n")
        # f.write(f"Black baseline CLIP score: {black_baseline_clip_score}\n")

    print("Saving stage 2 generations...")
    os.makedirs(os.path.join(out_dir,"svgs","unfixed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"svgs","pc_fixed"), exist_ok=True)
    os.makedirs(os.path.join(out_dir,"svgs","pi_fixed"), exist_ok=True)
    prompt_string = "\n".join(flattened_prompts)
    with open(os.path.join(out_dir,"svgs","prompts.txt"), "w") as f:
        f.write(prompt_string)
    for i, (bezier_points, positions) in tqdm(enumerate(generated_svgs), total=min(len(generated_svgs),max_num_svgs)):
        if i >= max_num_svgs:
            break
        save_svg(tokenizer, bezier_points, positions, padded_individual_max_length, stroke_width, os.path.join(out_dir,"svgs","unfixed", f"unfixed_{i}.svg"), num_strokes_to_paint=vq_context)
        save_svg(tokenizer, bezier_points, positions, padded_individual_max_length, stroke_width, os.path.join(out_dir,"svgs","pc_fixed", f"pc_fixed_{i}.svg"), num_strokes_to_paint=vq_context, fixing_method="min_dist_clip")
        save_svg(tokenizer, bezier_points, positions, padded_individual_max_length, stroke_width, os.path.join(out_dir,"svgs", "pi_fixed",f"pi_fixed_{i}.svg"), num_strokes_to_paint=vq_context, fixing_method="min_dist_interpolate")

    max_single_image = min(100, len(unfixed_renderings))
    save_image(make_grid(unfixed_renderings[:max_single_image], nrow=int(ceil(sqrt(max_single_image)))), os.path.join(out_dir,"unfixed_renderings.png"))
    save_image(make_grid(pc_fixed_renderings[:max_single_image], nrow=int(ceil(sqrt(max_single_image)))), os.path.join(out_dir,"pc_fixed_renderings.png"))
    save_image(make_grid(pi_fixed_renderings[:max_single_image], nrow=int(ceil(sqrt(max_single_image)))), os.path.join(out_dir,"pi_fixed_renderings.png"))

@torch.no_grad()
def benchmark_vsq_stage1(out_base_dir,
                         config_path,
                         ckpt_path,
                         num_samples,
                         max_num_svgs,
                         device,
                         clip_model = "openai/clip-vit-base-patch32",
                         **kwargs):
    print("received unused arguments: ", kwargs)
    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    if "wandb_version" in config.keys():
        config = map_wand_config(config)

    model = Vector_VQVAE(**config['model_params']).to(device).eval()

    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})

    dm = GlyphazznStage1Datamodule(**config['data_params'])
    dm.setup(stage="test")
    dataset = dm.test_dataset
    stroke_scale_factor = (dataset.individual_max_length+2) * config["data_params"]["stroke_width"] / 72
    random.seed(42)
    sample_idxs = random.sample(range(len(dataset)), num_samples)

    svg_save_dir = os.path.join(out_base_dir, "svgs")
    for subfolder in ["gt", "recons", "pi", "pc"]:
        os.makedirs(os.path.join(svg_save_dir, subfolder), exist_ok=True)

    with torch.no_grad():
        all_gt_drawings = []
        all_recons_drawings = []
        all_pi_drawings = []
        all_pc_drawings = []
        descriptions = []
        for idx in tqdm(sample_idxs):
            w = 480
            gt_drawing = dataset._get_full_svg_drawing(idx, width=w, as_tensor=False)

            # Reconstruct the SVG drawing
            patches, labels, positions, description = dataset._get_full_item(idx)
            descriptions.append(description)
            patches = patches.to(device)
            positions = positions.to(device)
            recons_drawing, _, shapes, stroke_width_predictions = model.reconstruct(patches, positions, dataset.individual_max_length +2, dataset.stroke_width, rendered_w=w, return_shapes=True)
            stroke_width_predictions = (stroke_width_predictions * stroke_scale_factor).flatten().tolist()
            # FIXME use real stroke width preds instead of dataset.stroke_width, this might be happening as diffvg uses stroke width differently
            stroke_width_predictions = [dataset.stroke_width] * len(stroke_width_predictions)
            pi_fixed_pos = min_dist_fix(shapes.detach().cpu(), method="min_dist_interpolate", max_dist=4.5)
            extra_strokes = [np.mean(stroke_width_predictions)] * (len(pi_fixed_pos) - len(shapes))
            pi_drawing = shapes_to_drawing(pi_fixed_pos, stroke_width_predictions + extra_strokes, w=w)
            pc_fixed_pos = min_dist_fix(shapes.detach().cpu(), method="min_dist_clip", max_dist=4.5)
            pc_drawing = shapes_to_drawing(pc_fixed_pos, stroke_width_predictions, w=w)
            all_gt_drawings.append(gt_drawing)
            all_recons_drawings.append(recons_drawing)
            all_pi_drawings.append(pi_drawing)
            all_pc_drawings.append(pc_drawing)

    print("Saving svgs...")
    for i, idx in tqdm(enumerate(sample_idxs[:max_num_svgs]), total=min(max_num_svgs, num_samples)):
        # save all svgs
        all_gt_drawings[i].saveas(os.path.join(svg_save_dir,"gt",f"gt_drawing_{idx}.svg"), pretty=True)
        all_recons_drawings[i].saveas(os.path.join(svg_save_dir,"recons",f"recons_drawing_{idx}.svg"), pretty=True)
        all_pi_drawings[i].saveas(os.path.join(svg_save_dir,"pi",f"pi_drawing_{idx}.svg"), pretty=True)
        all_pc_drawings[i].saveas(os.path.join(svg_save_dir,"pc",f"pc_drawing_{idx}.svg"), pretty=True)

    print("Computing FID score...")
    all_gt_rasters = [drawing_to_tensor(d) for d in all_gt_drawings]
    all_recons_rasters = [drawing_to_tensor(d) for d in all_recons_drawings]
    all_pi_rasters = [drawing_to_tensor(d) for d in all_pi_drawings]
    all_pc_rasters = [drawing_to_tensor(d) for d in all_pc_drawings]

    fid_recons = compute_fid_score(all_recons_rasters, all_gt_rasters, device, model_str=clip_model)
    fid_pi = compute_fid_score(all_pi_rasters, all_gt_rasters, device, model_str=clip_model)
    fid_pc = compute_fid_score(all_pc_rasters, all_gt_rasters, device, model_str=clip_model)

    print(f"FID recons: {fid_recons}")
    print(f"FID pi: {fid_pi}")
    print(f"FID pc: {fid_pc}")

    with open(os.path.join(out_base_dir, "results_fid.txt"), "w+") as f:
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"FID recons: {fid_recons}\n")
        f.write(f"FID pi: {fid_pi}\n")
        f.write(f"FID pc: {fid_pc}\n")

    print("Computing CLIP score...")
    clip_score_gt = compute_clip_score(all_gt_rasters, descriptions, device, model_str=clip_model)
    clip_score_recons = compute_clip_score(all_recons_rasters, descriptions, device, model_str=clip_model)
    clip_score_pi = compute_clip_score(all_pi_rasters, descriptions, device, model_str=clip_model)
    clip_score_pc = compute_clip_score(all_pc_rasters, descriptions, device, model_str=clip_model)

    print(f"CLIP score gt: {clip_score_gt}")
    print(f"CLIP score recons: {clip_score_recons}")
    print(f"CLIP score pi: {clip_score_pi}")
    print(f"CLIP score pc: {clip_score_pc}")

    with open(os.path.join(out_base_dir, "results_clip.txt"), "w+") as f:
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"CLIP score gt: {clip_score_gt}\n")
        f.write(f"CLIP score recons: {clip_score_recons}\n")
        f.write(f"CLIP score pi: {clip_score_pi}\n")
        f.write(f"CLIP score pc: {clip_score_pc}\n")

    with open(os.path.join(out_base_dir, "descriptions.txt"), "w+", encoding="utf-8") as f:
        f.write("\n".join(descriptions))

    print("Computing MSE...")
    mse_recons = torch.nn.functional.mse_loss(torch.stack(all_recons_rasters), torch.stack(all_gt_rasters)).item()
    mse_pi = torch.nn.functional.mse_loss(torch.stack(all_pi_rasters), torch.stack(all_gt_rasters)).item()
    mse_pc = torch.nn.functional.mse_loss(torch.stack(all_pc_rasters), torch.stack(all_gt_rasters)).item()

    print(f"MSE recons: {mse_recons}")
    print(f"MSE pi: {mse_pi}")
    print(f"MSE pc: {mse_pc}")

    with open(os.path.join(out_base_dir, "results_mse.txt"), "w+") as f:
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"MSE recons: {mse_recons}\n")
        f.write(f"MSE pi: {mse_pi}\n")
        f.write(f"MSE pc: {mse_pc}\n")

def main(eval_config_path, override:bool=False):
    random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(eval_config_path, 'r') as file:
        try:
            eval_config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    print(f"Found {len(eval_config)} eval configurations: {list(eval_config.keys())}")
    for k,v in eval_config.items():
        if v["type"].lower() == "stage2":
            print(f"Running eval on {k}...")
            if not os.path.exists(v["out_base_dir"]):
                os.makedirs(v["out_base_dir"])
            else:
                if not override and os.listdir(v["out_base_dir"]) > 0:
                    print(f"Warning: {v['out_base_dir']} already exists. Skipping...")
                    continue
                else:
                    print(f"Warning: {v['out_base_dir']} already exists. Overwriting files...")
            with open(os.path.join(v["out_base_dir"],"config.yaml"), 'w+', encoding="utf-8") as file:
                yaml.dump(v, file)

            for vq_context in tqdm(v["vq_contexts"]):
                out_dir = os.path.join(v["out_base_dir"],f"vq_context_{vq_context}_t0")
                benchmark_stage2_sgamo(**v,
                                        vq_context = vq_context,
                                       out_dir=out_dir,
                                       device=device)
                gc.collect()
                torch.cuda.empty_cache()
        elif v["type"].lower() == "stage1" or v["type"].lower() == "vsq":
            print(f"Running eval on {k}...")
            if not os.path.exists(v["out_base_dir"]):
                os.makedirs(v["out_base_dir"])
            else:
                if not override and os.listdir(v["out_base_dir"]) > 0:
                    print(f"Warning: {v['out_base_dir']} already exists. Skipping...")
                    continue
                else:
                    print(f"Warning: {v['out_base_dir']} already exists. Overwriting files...")
            with open(os.path.join(v["out_base_dir"],"config.yaml"), 'w+', encoding="utf-8") as file:
                yaml.dump(v, file)

            benchmark_vsq_stage1(**v, device = device)
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to the evaluation config yaml file", required=True)  # "configs/eval.yaml"
    parser.add_argument("--override", help="Override existing output dirs", action="store_true")
    args = parser.parse_args()

    main(args.config, override=args.override)