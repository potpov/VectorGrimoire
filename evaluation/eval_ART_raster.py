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
from torchvision.utils import make_grid
torch.cuda.is_available()
from utils import drawing_to_tensor
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
    parser.add_argument('--delay', type=int, dest="delay", help='time to sleep in seconds before execution', default=1)

    args = parser.parse_args()

    # CONFIG EVERYTHING HERE
    DEBUGGING = True if args.debug else False
    STAGE2_BASE_PATH = "/raid/marco.cipriano/results/svg/Grimoire/ART/ART_MNIST_BW_P6T0.2"
    NUM_SAMPLES = 3000 if not DEBUGGING else 50
    NUM_CONTEXT_PATCHES = [0, 3, 9]


    # ---------------------
    BATCH_SIZE_GENERATIONS = 16 if not DEBUGGING else 2
    CLIP_MODEL = "openai/clip-vit-base-patch32"
    GLOBAL_STROKE_WIDTH = 0.7
    NUM_REAL_IMAGES = 2300 if not DEBUGGING else 50
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
        "NUM_REAL_IMAGES": NUM_REAL_IMAGES,
        "SEED": SEED,
        "TEMPERATURE": TEMPERATURE,
        "NUM_SVGS_TO_SAVE": NUM_SVGS_TO_SAVE,
        "SAMPLING_METHOD": SAMPLING_METHOD,
        "MAX_DIST_FRAC": MAX_DIST_FRAC,
        "FIXING_METHODS": FIXING_METHODS,
        "GLOBAL_STROKE_WIDTH": GLOBAL_STROKE_WIDTH,

        "MNIST_FILTER_TH": 0.2,
        "MNIST_PNG_DIR": "/raid/marco.cipriano/data/SVG/Grimoire/MNIST/mnist_png",
        "MNIST_SUBSET": None,
        "VSQ_MNIST_TILES": 6,
        "VSQ_MNIST_PATCH_SIZE": 128,
    }

    time.sleep(args.delay)
    split = "test"
    BASE_OUT_DIR = os.path.join(
        STAGE2_BASE_PATH,
        "benchmark",
        ("full" if settings["MNIST_SUBSET"] is None else settings["MNIST_SUBSET"])
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
        subset=settings["MNIST_SUBSET"],
        val_BATCH_SIZE_GENERATIONS=NUM_REAL_IMAGES
    )
    unpatched_MNIST = MNISTDataset(
        data_path=settings["MNIST_PNG_DIR"],
        patch_size=128,
        num_tiles_per_row=1,
        subset=settings["MNIST_SUBSET"],
        val_BATCH_SIZE_GENERATIONS=NUM_REAL_IMAGES
    )

    encoded_dataset = VQDataModule(
        tokenizer=tokenizer,
        **config["data_params"],
        context_length=config['model_params']['max_seq_len'],
        train=False
    )
    encoded_dataset.setup()
    encoded_dataset.test_BATCH_SIZE_GENERATIONS = BATCH_SIZE_GENERATIONS
    encoded_dataset = encoded_dataset.test_dataloader()

    Grimoire_MNIST.setup()
    unpatched_MNIST.setup()

    Grimoire_MNIST = Grimoire_MNIST.test_dataloader()
    unpatched_MNIST = unpatched_MNIST.test_dataloader()


    ###########
    # RECON PIPELINE

    print("Loading reference dataset...")
    dataset_size = len(Grimoire_MNIST.dataset.labels)
    sampled_ids = random.sample(range(dataset_size), min(NUM_REAL_IMAGES, dataset_size))
    batches = [Grimoire_MNIST.dataset.__getitem__(idx) for idx in sampled_ids]

    curr_out_dir = os.path.join(BASE_OUT_DIR, f"recons")
    curr_svg_out_dir = os.path.join(curr_out_dir, "svgs")
    curr_gt_out_dir = os.path.join(curr_out_dir, "gt")
    os.makedirs(curr_svg_out_dir, exist_ok=True)
    os.makedirs(curr_gt_out_dir, exist_ok=True)
    rendered_generations = []
    rendered_gt = []
    print("Encoding - Deconding patches with the VSQ...")
    for idx in tqdm(range(NUM_REAL_IMAGES), total=NUM_REAL_IMAGES, leave=True):
        imgs, label, _, description = batches[idx]
        # tokenize the patches
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

        # rasterize the reconstructions
        rendered_generations.append(drawing_to_tensor(drawing))
        # retrieve the ground truth from the other dataset that does not patch
        unpatched_gt = unpatched_MNIST.dataset.__getitem__(sampled_ids[idx])[0].squeeze()
        rendered_gt.append(unpatched_gt)

        # save some svgs for the paper
        if idx < NUM_SAMPLES:
            drawing.saveas(os.path.join(curr_svg_out_dir, f"{idx}.svg"))
            save_image(unpatched_gt, os.path.join(curr_gt_out_dir, f"{idx}.png"))
            with open(os.path.join(curr_svg_out_dir, f"{idx}.txt"), "w") as f:
                f.write(description[0])

    print("Computing FID...")
    unfixed_fid_score = compute_fid_score(rendered_generations, rendered_gt, device, model_str=CLIP_MODEL)
    white_baseline_fid_score = compute_fid_score(
        list(torch.full(size=(len(rendered_gt), 3, RENDER_WIDTH, RENDER_WIDTH), fill_value=0.99999)),
        rendered_gt,
        device,
        model_str=CLIP_MODEL
    )
    black_baseline_fid_score = compute_fid_score(
        list(torch.full(size=(len(rendered_gt), 3, RENDER_WIDTH, RENDER_WIDTH), fill_value=0.000001)),
        rendered_gt,
        device,
        model_str=CLIP_MODEL
    )
    mse = torch.nn.functional.mse_loss(
        torch.stack(rendered_generations),
        torch.stack(rendered_gt)
    )
    results_json = {
        "metrics_sample_size": len(rendered_generations),
        "mse": get_item_if_tensor(mse),
        "unfixed_fid": unfixed_fid_score.cpu().item(),
        "white_baseline_fid": white_baseline_fid_score.cpu().item(),
        "black_baseline_fid": black_baseline_fid_score.cpu().item(),
    }

    with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
        json.dump(results_json, f)


    ###########
    # GENERATION PIPELINE
    for NUM_CONTEXT_PATCHES in tqdm(NUM_CONTEXT_PATCHES, leave=True):
        print(f"NUM_CONTEXT_PATCHES: {NUM_CONTEXT_PATCHES}")
        vq_context = int(NUM_CONTEXT_PATCHES * (art.ncode + 1))
        curr_out_dir = os.path.join(BASE_OUT_DIR, f"stroke_context_{NUM_CONTEXT_PATCHES}")

        if os.path.exists(os.path.join(curr_out_dir, "results.json")):
            print(f"[INFO] NUM_CONTEXT_PATCHES {NUM_CONTEXT_PATCHES} already exists, Skipping...")
            continue

        curr_svg_out_dir = os.path.join(curr_out_dir, "svgs")
        os.makedirs(curr_svg_out_dir, exist_ok=True)

        generations = []
        captions = []
        print(f"Generating...")
        for batch_id, (text_tokens, attention_mask, vq_tokens, _, _) in tqdm(enumerate(encoded_dataset), total=math.ceil(NUM_SAMPLES / BATCH_SIZE_GENERATIONS)):
            # eos_token = stage_2_model.special_token_mapping.get("<EOS>")
            text_tokens = text_tokens.to(device)
            attention_mask = attention_mask.to(device)
            curr_vq_tokens = vq_tokens[:, :vq_context + 1].clone().to(device)
            generation, reason = art.generate(text_tokens, attention_mask, curr_vq_tokens,
                                                        temperature=TEMPERATURE, sampling_method=SAMPLING_METHOD)
            generations.extend([g for g in generation])
            captions.extend([tokenizer.decode_text(tok) for tok in text_tokens])
            if len(generations) >= NUM_SAMPLES:
                break

        print("Saving some generations...")
        for i, g in tqdm(enumerate(generations[:NUM_SVGS_TO_SAVE]), total=NUM_SVGS_TO_SAVE):
            num_strokes_to_paint = NUM_CONTEXT_PATCHES
            tokenizer.use_text_encoder_only = False
            drawing = tokenizer._tokens_to_svg_drawing(
                g,
                only_patch_tokens=False,  # we predict positions
                w=RENDER_WIDTH
            )
            drawing.saveas(os.path.join(curr_svg_out_dir, f"{i}.svg"))
            caption = captions[i]
            with open(os.path.join(curr_svg_out_dir, f"{i}.txt"), "w") as f:
                f.write(caption)

        print("Rendering generations...")
        rendered_generations = []
        for g in tqdm(generations, total=len(generations)):
            tokenizer.use_text_encoder_only = False
            drawing = tokenizer._tokens_to_svg_drawing(
                g,
                only_patch_tokens=False,  # we predict positions
                w=RENDER_WIDTH
            )
            rendered_generations.append(drawing_to_tensor(drawing))

        ########
        # METRICS COMPUTATION

        # Randomly sampling some images from MNIST with no patching
        print("Loading reference dataset...")
        real_imgs = [
            unpatched_MNIST.dataset.__getitem__(idx)[0].squeeze()
            for idx in random.sample(
                range(len(unpatched_MNIST) * NUM_REAL_IMAGES),
                min(NUM_REAL_IMAGES, len(unpatched_MNIST)))
        ]


        print("Computing FID...")
        unfixed_fid_score = compute_fid_score(rendered_generations, real_imgs, device, model_str=CLIP_MODEL)
        white_baseline_fid_score = compute_fid_score(
            list(torch.full(size=(len(real_imgs), 3, RENDER_WIDTH, RENDER_WIDTH), fill_value=0.99999)),
            real_imgs,
            device,
            model_str=CLIP_MODEL
        )
        black_baseline_fid_score = compute_fid_score(
            list(torch.full(size=(len(real_imgs), 3, RENDER_WIDTH, RENDER_WIDTH), fill_value=0.000001)),
            real_imgs,
            device,
            model_str=CLIP_MODEL
        )

        # save some aggregated images to get a feel
        real_imgs_grid = make_grid(real_imgs[:10], nrow=10)
        unfixed_grid = make_grid(rendered_generations[:10], nrow=10)
        [save_image(x, os.path.join(curr_out_dir, f"{name}.png")) for x, name in
         zip([real_imgs_grid, unfixed_grid],
             ["real_imgs", "unfixed"])]

        print(f"Unfixed FID: {unfixed_fid_score}")
        print(f"White baseline FID: {white_baseline_fid_score}")
        print(f"Black baseline FID: {black_baseline_fid_score}")

        results_json = {
            "unfixed_fid": unfixed_fid_score.cpu().item(),
            "white_baseline_fid": white_baseline_fid_score.cpu().item(),
            "black_baseline_fid": black_baseline_fid_score.cpu().item(),
        }

        with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
            json.dump(results_json, f)

        print("Computing CLIP score...")
        adjusted_prompts = [f"{str(c)} in black color" for c in captions]

        prompt_string = "\n".join(captions)
        clip_adjusted_prompt_string = "\n".join(adjusted_prompts)
        with open(os.path.join(BASE_OUT_DIR, "prompts.txt"), "w") as f:
            f.write(prompt_string)
        with open(os.path.join(BASE_OUT_DIR, "clip_adjusted_prompts.txt"), "w") as f:
            f.write(clip_adjusted_prompt_string)

        clip_score = -1  # compute_clip_score(rendered_generations_unfixed, captions, device, model_str = CLIP_MODEL)
        prompt_adjusted_clip_score = compute_clip_score(
            rendered_generations,
            adjusted_prompts,
            device,
            model_str=CLIP_MODEL
        )

        white_baseline_clip_score = -1
        black_baseline_clip_score = -1
        prompt_adjusted_white_baseline_clip_score = compute_clip_score(
            list(torch.full(size=(len(adjusted_prompts), 3, RENDER_WIDTH, RENDER_WIDTH), fill_value=0.99999)),
            adjusted_prompts,
            device,
            model_str=CLIP_MODEL
        )
        prompt_adjusted_black_baseline_clip_score = compute_clip_score(
            list(torch.full(size=(len(adjusted_prompts), 3, RENDER_WIDTH, RENDER_WIDTH), fill_value=0.000001)),
            adjusted_prompts,
            device,
            model_str=CLIP_MODEL
        )

        results_json = {
            "generation_sample_size": len(rendered_generations),
            "unfixed_clip": get_item_if_tensor(clip_score),
            "prompt_adjusted_unfixed_clip": get_item_if_tensor(prompt_adjusted_clip_score),
            "white_baseline_clip": get_item_if_tensor(white_baseline_clip_score),
            "black_baseline_clip": get_item_if_tensor(black_baseline_clip_score),
            "prompt_adjusted_white_baseline_clip": get_item_if_tensor(prompt_adjusted_white_baseline_clip_score),
            "prompt_adjusted_black_baseline_clip": get_item_if_tensor(prompt_adjusted_black_baseline_clip_score),
            **results_json
        }

        with open(os.path.join(curr_out_dir, "results.json"), "w") as f:
            json.dump(results_json, f)

    print("DONE.")
