# TODO: THIS SCRIPT HAS TO BE CHECKED!!
import json
import math
import os
from typing import List
import yaml
from models import VQ_SVG_Stage2, VSQ
from tokenizer import VQTokenizer
from experiment import SVG_VQVAE_Stage2_Experiment
import torch
import pandas as pd
import random
import matplotlib.pyplot as plt
import time
import numpy as np
import torchvision.utils as vutils
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from dataset import GenericRasterizedSVGDataset, VSQDatamodule, VQDataModule, VSQDataset
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
from glob import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_item_if_tensor(potential_tensor):
    if isinstance(potential_tensor, Tensor):
        return potential_tensor.cpu().detach().item()
    else:
        return potential_tensor


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
def compute_fid_score(generated_images, real_images, device, model_str: str = "openai/clip-vit-base-patch32"):
    # print(f"Computing FID with model {model_str} on device {device}")
    model = CLIPModel.from_pretrained(model_str)
    processor = AutoProcessor.from_pretrained(model_str)
    wrapper = CLIPWrapper(model, processor, device)
    fid = FrechetInceptionDistance(feature=wrapper, normalize=True)
    fid = fid.to(device)
    bs = 32
    # print("Adding generated images...")
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i + bs]).to(device)
        fid.update(generated_images_batch, real=False)
    # print("Adding real images...")
    for i in tqdm(range(0, len(real_images), bs)):
        real_images_batch = torch.stack(real_images[i:i + bs]).to(device)
        fid.update(real_images_batch, real=True)

    return fid.compute()


@torch.no_grad()
def compute_clip_score(generated_images: List, captions: List, device, model_str: str = "openai/clip-vit-base-patch32",
                       do_rescale=False):
    # print(f"Computing CLIP score with model {model_str} on device {device}")
    metric = CLIPScore(model_name_or_path=model_str)
    metric = metric.to(device)
    bs = 32
    for i in tqdm(range(0, len(generated_images), bs)):
        generated_images_batch = torch.stack(generated_images[i:i + bs]).to(device)
        captions_batch = captions[i:i + bs]
        metric.update(generated_images_batch, captions_batch, do_rescale=do_rescale)

    return metric.compute()


def load_model_from_basepath(basepath, device="cpu"):
    """
    returns model, ds, config
    """
    config = yaml.load(open(os.path.join(basepath, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    config["data_params"]["max_shapes_per_svg"] = 2000
    config["data_params"]["train_batch_size"] = 2
    config["data_params"]["val_batch_size"] = 2
    model = VSQ(**config["model_params"]).to(device)
    all_ckpts = glob(os.path.join(basepath, "checkpoints", "*.ckpt"))
    # sort by date
    latest_ckpt_path = sorted(all_ckpts, key=os.path.getmtime)[-1]
    state_dict = torch.load(latest_ckpt_path, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    ds = VSQDataset(**config["data_params"], train=False)
    model = model.eval()
    return model, ds, config


def load_stage2_model_from_basepath(vsq_model, basepath, device="cpu"):
    """
    returns model, ds, config
    """
    config = yaml.load(open(os.path.join(basepath, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    # config = map_wand_config(config)
    config["data_params"]["fraction_of_class_only_inputs"] = 0.0
    config["data_params"]["fraction_of_blank_inputs"] = 0.0
    config["data_params"]["fraction_of_iconshop_chatgpt_inputs"] = 0.3
    config["data_params"]["fraction_of_full_description_inputs"] = 0.7

    tokenizer = VQTokenizer(vsq_model,
                            config["data_params"].get("grid_size") or config["data_params"].get("width"),
                            config['stage1_params']["num_codes_per_shape"],
                            config["model_params"]["text_encoder_str"],
                            lseg=config["stage1_params"]["lseg"],
                            device=device,
                            max_text_token_length=config["data_params"].get("max_text_token_length") or 50)

    config["model_params"].pop("name")
    model = VQ_SVG_Stage2(tokenizer, **config["model_params"], device=device)

    text_only_tokenizer = VQTokenizer(vsq_model,
                                      config["data_params"].get("grid_size") or config["data_params"].get("width"),
                                      config['stage1_params']["num_codes_per_shape"],
                                      config["model_params"]["text_encoder_str"],
                                      use_text_encoder_only=True,
                                      lseg=config["stage1_params"]["lseg"],
                                      codebook_size=tokenizer.codebook_size,
                                      max_text_token_length=config["data_params"].get("max_text_token_length") or 50, )
    dm = VQDataModule(tokenizer=text_only_tokenizer,
                      **config["data_params"],
                      context_length=config['model_params']['max_seq_len'],
                      train=False)
    # dm.setup(return_ids=True)
    dm.setup()
    for ds in [dm.train_dataset, dm.val_dataset, dm.test_dataset]:
        ds.fraction_of_class_only_inputs = config["data_params"]["fraction_of_class_only_inputs"]
        ds.fraction_of_blank_inputs = config["data_params"]["fraction_of_blank_inputs"]
        ds.fraction_of_iconshop_chatgpt_inputs = config["data_params"]["fraction_of_iconshop_chatgpt_inputs"]
        ds.fraction_of_full_description_inputs = config["data_params"]["fraction_of_full_description_inputs"]

    all_ckpts = glob(os.path.join(basepath, "checkpoints", "*.ckpt"))
    # filter out last and instead take lowest eval loss
    all_ckpts = [x for x in all_ckpts if not "last.ckpt" in x]
    # sort by date
    latest_ckpt_path = sorted(all_ckpts, key=os.path.getmtime)[-1]
    state_dict = torch.load(latest_ckpt_path, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict(
            {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()})
    model = model.eval()
    return model, dm, config



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c', dest="filename", metavar='FILE', help='path to the config file',
                        default='/raid/marco.cipriano/results/svg/Grimoire/ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid')
    parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')
    parser.add_argument('--delay', type=int, dest="delay", help='time to sleep in seconds before execution', default=1)

    args = parser.parse_args()

    # CONFIG EVERYTHING HERE
    DEBUGGING = True if args.debug else False
    STAGE2_BASE_PATH = args.filename
    # STAGE2_BASE_PATH = "/scratch2/moritz_logs/thesis/Stage2_figr8/nseg=4_ncode=2_lseg=5"
    NUM_SAMPLES = 3000 if not DEBUGGING else 50

    NUM_CONTEXT_STROKES = [0, 1, 3, 6, 10, 15, 20, 25, 30]
    # NUM_CONTEXT_STROKES = [0]


    BASE_PROMPT = "Black and white icon of {x}, vector graphic, outline"
    # BASE_OUT_DIR = os.path.join(STAGE2_BASE_PATH, "validation")
    # ---------------------
    BATCH_SIZE = 16 if not DEBUGGING else 2
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    RENDER_WIDTH = 480
    GLOBAL_STROKE_WIDTH = 0.7
    NUM_REAL_IMAGES = 2300 if not DEBUGGING else 50
    SEED = 42
    TEMPERATURE = 0.1
    NUM_SVGS_TO_SAVE = NUM_SAMPLES if not DEBUGGING else 5
    SAMPLING_METHOD = None
    MAX_DIST_FRAC = 4 / 72
    FIXING_METHODS = []
    # FIXING_METHODS = ["clip", "interpolate", "min_dist_clip", "min_dist_interpolate"]
    # ---------------------

    # save all of these settings above in a config file
    settings = {
        "STAGE2_BASE_PATH": STAGE2_BASE_PATH,
        "NUM_SAMPLES": NUM_SAMPLES,
        "NUM_CONTEXT_STROKES": NUM_CONTEXT_STROKES,
        "BASE_PROMPT": BASE_PROMPT,
        "BATCH_SIZE": BATCH_SIZE,
        "CLIP_MODEL": CLIP_MODEL,
        "RENDER_WIDTH": RENDER_WIDTH,
        "NUM_REAL_IMAGES": NUM_REAL_IMAGES,
        "SEED": SEED,
        "TEMPERATURE": TEMPERATURE,
        "NUM_SVGS_TO_SAVE": NUM_SVGS_TO_SAVE,
        "SAMPLING_METHOD": SAMPLING_METHOD,
        "MAX_DIST_FRAC": MAX_DIST_FRAC,
        "FIXING_METHODS": FIXING_METHODS,
        "GLOBAL_STROKE_WIDTH": GLOBAL_STROKE_WIDTH,
    }

    time.sleep(args.delay)
    # for split in ["validation", "test"]:
    for split in ["test"]:
        BASE_OUT_DIR = os.path.join(STAGE2_BASE_PATH, split)
        os.makedirs(BASE_OUT_DIR, exist_ok=True)

        with open(os.path.join(BASE_OUT_DIR, "config.yaml"), "w") as f:
            yaml.dump(settings, f)

        seed_everything(SEED)

        # load config to extract stage1 params
        config = yaml.load(open(os.path.join(STAGE2_BASE_PATH, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
        # config = map_wand_config(config)

        # load VSQ
        vsq_base_path = config["stage1_params"]["checkpoint_path"].split("checkpoints")[0]
        vsq_model = load_model_from_basepath(vsq_base_path, device=device)[0]

        # get model and data module
        stage_2_model, stage2_dm, stage2_config = load_stage2_model_from_basepath(vsq_model, STAGE2_BASE_PATH,
                                                                                  device=device)
        stage2_dm.test_batch_size = BATCH_SIZE

        # generation pipeline
        for num_context_strokes in tqdm(NUM_CONTEXT_STROKES, leave=True):
            print(f"num_context_strokes: {num_context_strokes}")
            if split == "validation":
                dl = stage2_dm.val_dataloader()
            else:
                dl = stage2_dm.test_dataloader()
            # dl = stage2_dm.test_dataloader()
            # dl = stage2_dm.val_dataloader()
            vq_context = int(num_context_strokes * (stage_2_model.ncode + 1))
            curr_out_dir = os.path.join(BASE_OUT_DIR, f"stroke_context_{num_context_strokes}")

            if os.path.exists(os.path.join(curr_out_dir, "results.json")):
                print(f"[INFO] num_context_strokes {num_context_strokes} already exists, Skipping...")
                continue

            curr_svg_out_dir = os.path.join(curr_out_dir, "svgs")
            os.makedirs(curr_svg_out_dir, exist_ok=True)
            for fixing_method in FIXING_METHODS:
                os.makedirs(os.path.join(curr_svg_out_dir, fixing_method), exist_ok=True)

            generations = []
            captions = []
            all_ids = []
            print(f"Generating {split} set...")
            for text_tokens, attention_mask, vq_tokens, _, _, svg_ids in tqdm(dl,
                                                                              total=math.ceil(NUM_SAMPLES / BATCH_SIZE)):
                # eos_token = stage_2_model.special_token_mapping.get("<EOS>")
                text_tokens = text_tokens.to(device)
                attention_mask = attention_mask.to(device)
                curr_vq_tokens = vq_tokens[:, :vq_context + 1].clone().to(device)
                generation, reason = stage_2_model.generate(text_tokens, attention_mask, curr_vq_tokens,
                                                            temperature=TEMPERATURE, sampling_method=SAMPLING_METHOD)
                # [stage_2_model.tokenizer._tokens_to_svg_drawing(g, global_stroke_width=GLOBAL_STROKE_WIDTH, post_process=False, num_strokes_to_paint=0) for g in generation]
                generations.extend([g for g in generation])
                captions.extend([stage_2_model.tokenizer.decode_text(tok) for tok in text_tokens])
                all_ids.extend([x for x in svg_ids])
                if len(generations) >= NUM_SAMPLES:
                    break

            print("Saving some generations...")
            for i, g in tqdm(enumerate(generations[:NUM_SVGS_TO_SAVE]), total=NUM_SVGS_TO_SAVE):
                num_strokes_to_paint = num_context_strokes
                drawing = stage_2_model.tokenizer._tokens_to_svg_drawing(g, global_stroke_width=GLOBAL_STROKE_WIDTH,
                                                                         post_process=False,
                                                                         num_strokes_to_paint=num_strokes_to_paint)
                drawing.saveas(os.path.join(curr_svg_out_dir, f"{all_ids[i]}.svg"))
                caption = captions[i]
                with open(os.path.join(curr_svg_out_dir, f"{all_ids[i]}.txt"), "w") as f:
                    f.write(caption)

                for fixing_method in FIXING_METHODS:
                    out_path = os.path.join(curr_svg_out_dir, fixing_method)
                    drawing_fixed = stage_2_model.tokenizer._tokens_to_svg_drawing(g,
                                                                                   global_stroke_width=GLOBAL_STROKE_WIDTH,
                                                                                   method=fixing_method,
                                                                                   num_strokes_to_paint=num_strokes_to_paint,
                                                                                   post_process=True, connect_last=False,
                                                                                   max_dist_frac=MAX_DIST_FRAC)
                    drawing_fixed.saveas(os.path.join(out_path, f"{all_ids[i]}_{fixing_method}_fixed.svg"))
                    with open(os.path.join(out_path, f"{all_ids[i]}_{fixing_method}_fixed.txt"), "w") as f:
                        f.write(caption)

            print("Rendering generations...")
            rendered_generations_unfixed = []
            rendered_generations_pc_fixed = []
            rendered_generations_pi_fixed = []
            for g in tqdm(generations, total=len(generations)):
                unfixed_drawing = stage_2_model.tokenizer._tokens_to_svg_drawing(g, post_process=False, w=RENDER_WIDTH,
                                                                                 global_stroke_width=GLOBAL_STROKE_WIDTH)
                pc_fixed_drawing = stage_2_model.tokenizer._tokens_to_svg_drawing(g, method="min_dist_clip",
                                                                                  post_process=True, w=RENDER_WIDTH,
                                                                                  max_dist_frac=MAX_DIST_FRAC,
                                                                                  global_stroke_width=GLOBAL_STROKE_WIDTH)
                pi_fixed_drawing = stage_2_model.tokenizer._tokens_to_svg_drawing(g, method="min_dist_interpolate",
                                                                                  post_process=True, w=RENDER_WIDTH,
                                                                                  max_dist_frac=MAX_DIST_FRAC,
                                                                                  global_stroke_width=GLOBAL_STROKE_WIDTH)

                rendered_generations_unfixed.append(drawing_to_tensor(unfixed_drawing))
                rendered_generations_pc_fixed.append(drawing_to_tensor(pc_fixed_drawing))
                rendered_generations_pi_fixed.append(drawing_to_tensor(pi_fixed_drawing))

            print("Loading reference dataset...")
            rasterized_ds = GenericRasterizedSVGDataset(config["data_params"]["csv_path"],
                                                        train=None,
                                                        fill=False,
                                                        img_size=480,
                                                        global_stroke_width=GLOBAL_STROKE_WIDTH,
                                                        subset=None)
            indices = random.sample(range(len(rasterized_ds)), min(NUM_REAL_IMAGES, len(rasterized_ds)))
            real_imgs = []
            for i in tqdm(indices):
                real_imgs.append(rasterized_ds[i][0])

            print("Computing FID...")
            unfixed_fid_score = compute_fid_score(rendered_generations_unfixed, real_imgs, device, model_str=CLIP_MODEL)
            pc_fixed_fid_score = compute_fid_score(rendered_generations_pc_fixed, real_imgs, device, model_str=CLIP_MODEL)
            pi_fixed_fid_score = compute_fid_score(rendered_generations_pi_fixed, real_imgs, device, model_str=CLIP_MODEL)
            white_baseline_fid_score = compute_fid_score(
                [torch.ones((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(real_imgs))], real_imgs,
                device, model_str=CLIP_MODEL)
            black_baseline_fid_score = compute_fid_score(
                [torch.zeros((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(real_imgs))], real_imgs,
                device, model_str=CLIP_MODEL)

            # save some aggregated images to get a feel
            real_imgs_grid = make_grid(real_imgs[:10], nrow=10)
            unfixed_grid = make_grid(rendered_generations_unfixed[:10], nrow=10)
            pc_fixed_grid = make_grid(rendered_generations_pc_fixed[:10], nrow=10)
            pi_fixed_grid = make_grid(rendered_generations_pi_fixed[:10], nrow=10)
            [save_image(x, os.path.join(curr_out_dir, f"{name}.png")) for x, name in
             zip([real_imgs_grid, unfixed_grid, pc_fixed_grid, pi_fixed_grid],
                 ["real_imgs", "unfixed", "pc_fixed", "pi_fixed"])]

            print(f"Unfixed FID: {unfixed_fid_score}")
            print(f"PC fixed FID: {pc_fixed_fid_score}")
            print(f"PI fixed FID: {pi_fixed_fid_score}")
            print(f"White baseline FID: {white_baseline_fid_score}")
            print(f"Black baseline FID: {black_baseline_fid_score}")

            results_json = {
                "unfixed_fid": unfixed_fid_score.cpu().item(),
                "pc_fixed_fid": pc_fixed_fid_score.cpu().item(),
                "pi_fixed_fid": pi_fixed_fid_score.cpu().item(),
                "white_baseline_fid": white_baseline_fid_score.cpu().item(),
                "black_baseline_fid": black_baseline_fid_score.cpu().item(),
            }

            with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
                json.dump(results_json, f)

            print("Computing CLIP score...")
            adjusted_prompts = [BASE_PROMPT.format(x=c) for c in captions]

            prompt_string = "\n".join(captions)
            clip_adjusted_prompt_string = "\n".join(adjusted_prompts)
            with open(os.path.join(BASE_OUT_DIR, "prompts.txt"), "w") as f:
                f.write(prompt_string)
            with open(os.path.join(BASE_OUT_DIR, "clip_adjusted_prompts.txt"), "w") as f:
                f.write(clip_adjusted_prompt_string)

            unfixed_clip_score = -1  # compute_clip_score(rendered_generations_unfixed, captions, device, model_str = CLIP_MODEL)
            pc_fixed_clip_score = -1  # compute_clip_score(rendered_generations_pc_fixed, captions, device, model_str = CLIP_MODEL)
            pi_fixed_clip_score = -1  # compute_clip_score(rendered_generations_pi_fixed, captions, device, model_str = CLIP_MODEL)
            prompt_adjusted_unfixed_clip_score = compute_clip_score(rendered_generations_unfixed, adjusted_prompts, device,
                                                                    model_str=CLIP_MODEL)
            prompt_adjusted_pc_fixed_clip_score = compute_clip_score(rendered_generations_pc_fixed, adjusted_prompts,
                                                                     device, model_str=CLIP_MODEL)
            prompt_adjusted_pi_fixed_clip_score = compute_clip_score(rendered_generations_pi_fixed, adjusted_prompts,
                                                                     device, model_str=CLIP_MODEL)

            white_baseline_clip_score = -1  # compute_clip_score([torch.ones((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(captions))], captions, device, model_str = CLIP_MODEL)
            black_baseline_clip_score = -1  # compute_clip_score([torch.zeros((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(captions))], captions, device, model_str = CLIP_MODEL)
            prompt_adjusted_white_baseline_clip_score = compute_clip_score(
                [torch.ones((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(adjusted_prompts))],
                adjusted_prompts, device, model_str=CLIP_MODEL)
            prompt_adjusted_black_baseline_clip_score = compute_clip_score(
                [torch.zeros((3, 480, 480), dtype=torch.float32, device=device) for _ in range(len(adjusted_prompts))],
                adjusted_prompts, device, model_str=CLIP_MODEL)

            results_json = {
                "generation_sample_size": len(rendered_generations_unfixed),
                "unfixed_clip": get_item_if_tensor(unfixed_clip_score),
                "pc_fixed_clip": get_item_if_tensor(pc_fixed_clip_score),
                "pi_fixed_clip": get_item_if_tensor(pi_fixed_clip_score),
                "prompt_adjusted_unfixed_clip": get_item_if_tensor(prompt_adjusted_unfixed_clip_score),
                "prompt_adjusted_pc_fixed_clip": get_item_if_tensor(prompt_adjusted_pc_fixed_clip_score),
                "prompt_adjusted_pi_fixed_clip": get_item_if_tensor(prompt_adjusted_pi_fixed_clip_score),
                "white_baseline_clip": get_item_if_tensor(white_baseline_clip_score),
                "black_baseline_clip": get_item_if_tensor(black_baseline_clip_score),
                "prompt_adjusted_white_baseline_clip": get_item_if_tensor(prompt_adjusted_white_baseline_clip_score),
                "prompt_adjusted_black_baseline_clip": get_item_if_tensor(prompt_adjusted_black_baseline_clip_score),
                **results_json
            }

            with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
                json.dump(results_json, f)

    print("DONE.")



# call me with those arguments:
# FIGR-8
# --config /raid/marco.cipriano/results/svg/Grimoire/ART/figr8/nseg=4_ncode=2_lseg=5_alpha01_256grid
# /raid/marco.cipriano/results/svg/Grimoire/VSQ/figr8/VSQ_nseg=4_ncode=2_lseg=5.0_alpha=0.1None
# FONTS
# --config