import math
from typing import List
from utils import get_filter_function
import numpy as np
from torch import Tensor
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore
from transformers import AutoProcessor, CLIPModel
from torch import nn
import time
import matplotlib.pyplot as plt
import random
from torchvision.utils import save_image
import os
import yaml
import argparse
from dataset import MNISTDataset
from dataset import VQDataModule
from models import VQ_SVG_Stage2, VSQ
from tokenizer import RasterVQTokenizer
import json
import torch
from torchvision.transforms import Resize
torch.cuda.is_available()
from utils import drawing_to_tensor
from glob import glob
from utils import svg_file_path_to_tensor
from torchvision.utils import make_grid
import re

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
        metric.update(generated_images_batch, captions_batch)

    return metric.compute()


def load_models(config, filter_fn, device="cpu"):
    """
    returns vq_model, tokenizer, art
    """
    #########
    # Load STAGE 1 model
    vqs_model = VSQ(patch_size=config['data_params']["patch_size"], **config['stage1_params'], device=device)
    state_dict = torch.load(config['stage1_params']["checkpoint_path"], map_location=device)["state_dict"]
    try:
        vqs_model.load_state_dict(state_dict)
    except:
        vqs_model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    vq_model = vqs_model.eval()

    tokenizer = RasterVQTokenizer(vqs_model,
                                  tokens_per_patch=1,
                                  do_tokenize_positions=False,
                                  patch_size=config['data_params']["patch_size"],
                                  num_tiles_per_row=config['data_params']["num_tiles_per_row"],
                                  device=device,
                                  use_text_encoder_only=False,
                                  filter_fn=filter_fn
                                  )
    #########
    # Load STAGE 2 model, tokenizer and dataset
    config["model_params"].pop("name")
    art = VQ_SVG_Stage2(tokenizer, **config['model_params'], device=device)

    all_ckpts = glob(os.path.join(config["logging_params"]["save_dir"], "checkpoints", "*.ckpt"))
    all_ckpts = [x for x in all_ckpts if not "last.ckpt" in x]
    latest_ckpt_path = sorted(all_ckpts, key=os.path.getmtime)[-1]
    state_dict = torch.load(latest_ckpt_path, map_location=device)["state_dict"]
    try:
        art.load_state_dict(state_dict)
    except:
        art.load_state_dict(
            {k.replace("model.", "", 1) if k.startswith("model.") else k: v for k, v in state_dict.items()})
    art = art.eval()

    return vq_model, tokenizer, art



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--debug', action='store_true', help='disable wandb logs, set workers to 0. (default false)')
    parser.add_argument('--subset', type=str, dest="subset", help='subset to test', required=True)

    args = parser.parse_args()

    # CONFIG EVERYTHING HERE
    DEBUGGING = True if args.debug else False
    # STAGE2_BASE_PATH = "/raid/marco.cipriano/results/svg/Grimoire/ART/ART_MNIST_BW_P6T0.2"
    STAGE2_BASE_PATH = "/raid/marco.cipriano/results/svg/Grimoire/ART/ART_MNIST_BW_P6T0.2"
    NUM_SAMPLES = 5000 if not DEBUGGING else 50
    NUM_CONTEXT_PATCHES = [0]


    # ---------------------
    BATCH_SIZE_GENERATIONS = 16 if not DEBUGGING else 2
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    GLOBAL_STROKE_WIDTH = 0.7
    SEED = 42
    TEMPERATURE = 0.1
    NUM_SVGS_TO_SAVE = NUM_SAMPLES if not DEBUGGING else 5
    SAMPLING_METHOD = None
    MAX_DIST_FRAC = 4 / 72
    FIXING_METHODS = []
    RENDER_WIDTH = 128  # let's use always 128, set the same in the im2Vec evaluation script, and in the settings below

    # ---------------------

    # save all of these settings above in a config file
    settings = {
        "STAGE2_BASE_PATH": STAGE2_BASE_PATH,
        "NUM_SAMPLES": NUM_SAMPLES,
        "NUM_CONTEXT_PATCHES": NUM_CONTEXT_PATCHES,
        "BATCH_SIZE_GENERATIONS": BATCH_SIZE_GENERATIONS,
        "CLIP_MODEL": CLIP_MODEL,
        "RENDER_WIDTH": RENDER_WIDTH,
        "SEED": SEED,
        "TEMPERATURE": TEMPERATURE,
        "NUM_SVGS_TO_SAVE": NUM_SVGS_TO_SAVE,
        "SAMPLING_METHOD": SAMPLING_METHOD,
        "MAX_DIST_FRAC": MAX_DIST_FRAC,
        "FIXING_METHODS": FIXING_METHODS,
        "GLOBAL_STROKE_WIDTH": GLOBAL_STROKE_WIDTH,

        "MNIST_FILTER_TH": 0.2,
        "MNIST_PNG_DIR": "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_png",
        "MNIST_SUBSET": str(args.subset),  # options:  "5", "0", "full"
        "VSQ_MNIST_TILES": 6,
        "VSQ_MNIST_PATCH_SIZE": 128,
    }

    time.sleep(1)
    split = "test"
    BASE_OUT_DIR = os.path.join(
        STAGE2_BASE_PATH,
        "benchmark",
        settings["MNIST_SUBSET"]
    )
    os.makedirs(BASE_OUT_DIR, exist_ok=True)

    with open(os.path.join(BASE_OUT_DIR, "config.yaml"), "w") as f:
        yaml.dump(settings, f)

    seed_everything(SEED)

    # load config to extract stage1 params
    config = yaml.load(open(os.path.join(STAGE2_BASE_PATH, 'config.yaml'), 'r'), Loader=yaml.FullLoader)
    # config = map_wand_config(config)

    # load all models, datasets and tokenizers
    filter_fn = None
    if settings["MNIST_FILTER_TH"] is not None:
        print("Filtering patches with less than ", settings["MNIST_FILTER_TH"], " non-white pixels")
        filter_fn = get_filter_function(settings["MNIST_FILTER_TH"], parse_patches=False)
    vq_model, tokenizer, art = load_models(config, filter_fn, device=device)

    Grimoire_MNIST = MNISTDataset(
        data_path=settings["MNIST_PNG_DIR"],
        patch_size=settings["VSQ_MNIST_PATCH_SIZE"],
        num_tiles_per_row=settings["VSQ_MNIST_TILES"],
        subset=None if settings["MNIST_SUBSET"] == "full" else settings["MNIST_SUBSET"],
    )
    unpatched_MNIST = MNISTDataset(
        data_path=settings["MNIST_PNG_DIR"],
        patch_size=128,
        num_tiles_per_row=1,
        subset=None if settings["MNIST_SUBSET"] == "full" else settings["MNIST_SUBSET"],
    )

    encoded_dataset = VQDataModule(
        tokenizer=tokenizer,
        **config["data_params"],
        context_length=config['model_params']['max_seq_len'],
        subset=None if settings["MNIST_SUBSET"] == "full" else settings["MNIST_SUBSET"],
        train=False
    )
    encoded_dataset.setup()
    encoded_dataset.test_BATCH_SIZE_GENERATIONS = BATCH_SIZE_GENERATIONS
    encoded_dataset = encoded_dataset.test_dataloader()

    Grimoire_MNIST.setup()
    unpatched_MNIST.setup()

    Grimoire_MNIST = Grimoire_MNIST.test_dataloader()
    unpatched_MNIST = unpatched_MNIST.test_dataloader()

    r = Resize((RENDER_WIDTH, RENDER_WIDTH))

    ###########
    # RECON PIPELINE
    print("Loading reference dataset...")
    dataset_size = len(Grimoire_MNIST.dataset.labels)
    sampled_ids = random.sample(range(dataset_size), min(NUM_SAMPLES, dataset_size))
    batches = [Grimoire_MNIST.dataset.__getitem__(idx) for idx in sampled_ids]
    print("USING SAMPLE SIZE: ", len(sampled_ids))

    curr_out_dir = os.path.join(BASE_OUT_DIR, f"recons")
    curr_svg_out_dir = os.path.join(curr_out_dir, "svgs")
    curr_gt_out_dir = os.path.join(curr_out_dir, "gt")
    os.makedirs(curr_svg_out_dir, exist_ok=True)
    os.makedirs(curr_gt_out_dir, exist_ok=True)
    rendered_recons = []
    rendered_gt = []
    labels = []
    print("Encoding - Deconding patches with the VSQ...")
    for idx in tqdm(range(len(sampled_ids)), total=len(sampled_ids)):
        # batches[idx] gives a single image, they are in the shape (Patch, 3, 128, 128).
        # so for labels the first class is repeated patchsize-time
        imgs, label, _, description = batches[idx]
        # tokenize the patches
        labels.append(str(label[0]))
        tokenizer.use_text_encoder_only = False
        imgs = imgs.to(device)
        _, _, vq_tokens, end_token = tokenizer.tokenize(
            imgs,
            text="",
            include_positions=True,
            return_np_uint16=True
        )
        # reconstruct with trained VSQ module as svg
        drawing = tokenizer._tokens_to_svg_drawing(
            torch.asarray(np.concatenate([vq_tokens, end_token])).int().to(device),
            only_patch_tokens=False,  # we predict positions
            w=RENDER_WIDTH
        )

        # saving svg
        filepath = os.path.join(curr_svg_out_dir, f"{idx}.svg")
        drawing.saveas(filepath)
        # loading and rendering with filling
        recon_renders_filled = svg_file_path_to_tensor(filepath, stroke_width=0.4, image_size=32, filling=True)

        recon_renders_filled = r(recon_renders_filled)
        rendered_recons.append(recon_renders_filled)

        unpatched_gt = unpatched_MNIST.dataset.__getitem__(sampled_ids[idx])[0].squeeze()
        # unpatched_gt = r(make_grid(imgs, nrow=6, padding=0))
        rendered_gt.append(unpatched_gt)


    print("Computing FID...")
    fid_score = compute_fid_score(rendered_recons, rendered_gt, device, model_str=CLIP_MODEL)
    print("FID: ", get_item_if_tensor(fid_score))

    mse = torch.nn.functional.mse_loss(
        torch.stack(rendered_recons),
        torch.stack(rendered_gt)
    )
    print("MSE: ", get_item_if_tensor(mse))

    print("Computing CLIP score...")
    adjusted_prompts = [f"{str(c)} in black color" for c in labels]
    prompt_adjusted_clip_score = compute_clip_score(
        rendered_recons,
        adjusted_prompts,
        device,
        model_str=CLIP_MODEL
    )
    print("CLIP: ", get_item_if_tensor(prompt_adjusted_clip_score))

    # LOGGING RECONS
    results_json = {
        "metrics_sample_size": len(rendered_recons),
        "MSE": get_item_if_tensor(mse),
        "FID": get_item_if_tensor(fid_score),
        "CLIP": get_item_if_tensor(prompt_adjusted_clip_score),
    }
    with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
        json.dump(results_json, f)
    with open(os.path.join(curr_out_dir, "results.txt"), "w") as f:
        f.write(f"metrics_sample_size: {len(rendered_recons)}\n")
        f.write(f"MSE: {get_item_if_tensor(mse)}\n")
        f.write(f"FID: {get_item_if_tensor(fid_score)}\n")
        f.write(f"CLIP: {get_item_if_tensor(prompt_adjusted_clip_score)}\n")

    ###########
    # GENERATION PIPELINE
    for curr_context_patch in tqdm(NUM_CONTEXT_PATCHES):

        print(f"NUM_CONTEXT_PATCHES: {curr_context_patch}")
        vq_context = int(curr_context_patch * (art.ncode + 1))
        curr_out_dir = os.path.join(BASE_OUT_DIR, f"stroke_context_{curr_context_patch}")

        if os.path.exists(os.path.join(curr_out_dir, "results.json")):
            print(f"[INFO] NUM_CONTEXT_PATCHES {curr_context_patch} already exists, Overriding...")

        curr_svg_out_dir = os.path.join(curr_out_dir, "svgs")
        os.makedirs(curr_svg_out_dir, exist_ok=True)

        generations = []
        captions = []
        print(f"Loading the full captions and tokens for the encoded dataset...")
        dataset_size = len(encoded_dataset.dataset.split)
        sampled_ids = random.sample(range(dataset_size), min(NUM_SAMPLES, dataset_size))
        batches = [encoded_dataset.dataset.__getitem__(idx) for idx in sampled_ids]
        print("Generating...")
        for sampled_idx in tqdm(range(len(sampled_ids)), total=len(sampled_ids)):
            text_tokens, attention_mask, vq_tokens, _, _ = batches[sampled_idx]
            text_tokens = text_tokens.to(device)
            attention_mask = attention_mask.to(device)
            curr_vq_tokens = vq_tokens[:vq_context + 1].clone().to(device)
            generation, reason = art.generate(text_tokens[None], attention_mask[None], curr_vq_tokens[None],temperature=TEMPERATURE, sampling_method=SAMPLING_METHOD)

            generations.append(generation[0])
            captions.append(tokenizer.decode_text(text_tokens))

        print("Saving all generations...")
        for i in tqdm(range(len(generations)), total=len(generations)):
            num_strokes_to_paint = curr_context_patch
            tokenizer.use_text_encoder_only = False
            drawing = tokenizer._tokens_to_svg_drawing(
                generations[i],
                only_patch_tokens=False,  # we predict positions
                w=RENDER_WIDTH
            )
            drawing.saveas(os.path.join(curr_svg_out_dir, f"{i}.svg"))
            caption = captions[i]
            with open(os.path.join(curr_svg_out_dir, f"{i}.txt"), "w") as f:
                f.write(caption)

        print("Rendering generations...")
        render_unfilled_gen = []
        render_filled_gen = []
        for i in tqdm(range(len(generations)), total=len(generations)):
            filepath = os.path.join(curr_svg_out_dir, f"{i}.svg")
            tmp = svg_file_path_to_tensor(filepath, stroke_width=0.4, image_size=32, filling=True)
            tmp = r(tmp)
            render_filled_gen.append(tmp)
            tmp = svg_file_path_to_tensor(filepath, stroke_width=0.4, image_size=32, filling=False)
            tmp = r(tmp)
            render_unfilled_gen.append(tmp)

        ########
        # METRICS COMPUTATION
        # Randomly sampling some images from MNIST with no patching
        print("Loading reference dataset for FID score...")
        dataset_size = len(unpatched_MNIST.dataset.labels)
        sampled_ids = random.sample(range(dataset_size), min(NUM_SAMPLES, dataset_size))
        real_imgs = batches = [unpatched_MNIST.dataset.__getitem__(idx)[0].squeeze() for idx in sampled_ids]

        print("Computing FID...")
        unfilled_fid_score = compute_fid_score(render_unfilled_gen, real_imgs, device, model_str=CLIP_MODEL)
        filled_fid_score = compute_fid_score(render_filled_gen, real_imgs, device, model_str=CLIP_MODEL)

        # save some aggregated images to get a feel
        real_imgs_grid = make_grid(real_imgs[:10], nrow=10)
        unfixed_grid = make_grid(render_filled_gen[:10], nrow=10)
        [save_image(x, os.path.join(curr_out_dir, f"{name}.png")) for x, name in
         zip([real_imgs_grid, unfixed_grid],
             ["real_imgs", "unfixed"])]

        print(f"FID (unfilled): {unfilled_fid_score}")
        print(f"FID (filled): {filled_fid_score}")

        print("Computing CLIP score...")
        # adjusted_prompts = [f"{str(c)} in black color" for c in captions]

        # prompt_string = "\n".join(captions)
        # clip_adjusted_prompt_string = "\n".join(adjusted_prompts)
        # with open(os.path.join(BASE_OUT_DIR, "prompts.txt"), "w") as f:
        #     f.write(prompt_string)
        # with open(os.path.join(BASE_OUT_DIR, "clip_adjusted_prompts.txt"), "w") as f:
        #     f.write(clip_adjusted_prompt_string)

        clip_filled = compute_clip_score(render_filled_gen, captions, device, model_str=CLIP_MODEL)
        clip_unfilled = compute_clip_score(render_unfilled_gen, captions, device, model_str=CLIP_MODEL)

        # LOGGING GENERATIONS
        results_json = {
            "generation_sample_size": len(render_filled_gen),
            "CLIP (unfilled)": get_item_if_tensor(clip_unfilled),
            "CLIP (filled)": get_item_if_tensor(clip_filled),
            "FID (unfilled)": get_item_if_tensor(unfilled_fid_score),
            "FID (filled)": get_item_if_tensor(filled_fid_score),
        }
        print(results_json)
        with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
            json.dump(results_json, f)

        with open(os.path.join(curr_out_dir, "results.txt"), "w") as f:
            f.write(f"CLIP (unfilled): {get_item_if_tensor(clip_unfilled)}"),
            f.write(f"CLIP (filled): {get_item_if_tensor(clip_filled)}"),
            f.write(f"FID (unfilled): {get_item_if_tensor(unfilled_fid_score)}"),
            f.write(f"FID (filled): {get_item_if_tensor(filled_fid_score)}"),

    print(f"DONE! find stuff in {curr_out_dir}")
