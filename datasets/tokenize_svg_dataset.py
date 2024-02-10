from thesis.tokenizer import VQTokenizer
from thesis.dataset import GlyphazznStage1Dataset
from thesis.models import Vector_VQVAE
import numpy as np
import torch
from glob import glob
import os
from tqdm import tqdm
import yaml
import pandas as pd
import time
from torch.utils.data import DataLoader

def mycollate(batch):
    imgs, labels, centers, descriptions = zip(*batch)
    return imgs, labels, centers, descriptions

def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    CONTEXT_LENGTH = 256
    MODEL_WEIGHTS_PATH = "/scratch2/moritz_logs/SVG_VQVAE/Stage1/glyphazzn_full_single_code/checkpoints/last.ckpt"
    OUT_DIR = "/scratch2/moritz_data/glyphazzn/tokenized"
    TOP_LEVEL_DIR = "/scratch2/moritz_data/glyphazzn/svgs_simplified"
    CONFIG_PATH = "/home/mfeuerpfeil/master/thesis/configs/SVG_VQVAE_Stage1.yaml"

    with open(CONFIG_PATH, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['data_params']["max_shapes_per_svg"] = 512  # more than context length of 1024 (at least one patch and one pos token per svg shape) we'll never do I think
        
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    print("Loading model..")
    start_time = time.time()
    model = Vector_VQVAE(**config['model_params']).to(device)
    state_dict = torch.load(MODEL_WEIGHTS_PATH, map_location=device)["state_dict"]
    try:
        model.load_state_dict(state_dict)
    except:
        model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()})
    model = model.eval()
    elapsed_time = time.time() - start_time
    print(f"Model loaded in {elapsed_time} seconds")

    start_time = time.time()
    tokenizer = VQTokenizer(model, config["data_params"]["width"], 1, "bert-base-uncased")
    elapsed_time = time.time() - start_time
    print(f"Tokenizer loaded in {elapsed_time} seconds")

    start_time = time.time()
    print("Loading dataset..")

    ds_train = GlyphazznStage1Dataset(train=True, **config['data_params'])
    ds_test = GlyphazznStage1Dataset(train=False, **config['data_params'])
    dl_train = DataLoader(ds_train,
        batch_size=16,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )
    dl_test = DataLoader(ds_test,
        batch_size=16,
        num_workers=16,
        shuffle=False,
        pin_memory=False,
        collate_fn=mycollate,
    )

    elapsed_time = time.time() - start_time
    print(f"Datasets loaded in {elapsed_time} seconds")
    print("Number of Tokens: ",tokenizer.num_tokens)

    train_vq_tokens = []
    train_text_tokens = []
    get_item_times = []
    tokenization_times = []
    print("Processing training set..")
    for i, batch in tqdm(enumerate(dl_train), total=len(dl_train)):
        start_time = time.time()
        imgs, labels, centers, descriptions = batch
        elapsed_time = time.time() - start_time
        get_item_times.append(elapsed_time)
        start_time = time.time()
        for img, label, center, description in zip(imgs, labels, centers, descriptions):
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(img.to(device), center, text=description, return_np_uint16=True)
            train_vq_tokens.append(vq_tokens)
            train_text_tokens.append(text_tokens)
        elapsed_time = time.time() - start_time
        tokenization_times.append(elapsed_time)
        if i % 100 == 0 and i < 1000:
            print(f"Average dataloader time: {round(sum(get_item_times) / len(get_item_times), 2)} seconds")
            print(f"Average tokenization time: {round(sum(tokenization_times) / len(tokenization_times), 2)} seconds")
        if i % 200 == 0:
            np.save(os.path.join(OUT_DIR, "train_vq_tokens.npy"), np.concatenate(train_vq_tokens))
            np.save(os.path.join(OUT_DIR, "train_text_tokens.npy"), np.concatenate(train_text_tokens))
    train_vq_tokens = np.concatenate(train_vq_tokens)
    train_text_tokens = np.concatenate(train_text_tokens)
    np.save(os.path.join(OUT_DIR, "train_vq_tokens.npy"), train_vq_tokens)
    np.save(os.path.join(OUT_DIR, "train_text_tokens.npy"), train_text_tokens)

    test_vq_tokens = []
    test_text_tokens = []
    print("Processing test set..")
    for i, batch in tqdm(enumerate(dl_train), total=len(dl_test)):
        imgs, labels, centers, descriptions = batch
        for img, label, center, description in zip(imgs, labels, centers, descriptions):
            start_token, text_tokens, vq_tokens, end_token = tokenizer.tokenize(img, center, text=description, return_np_uint16=True)
            test_vq_tokens.append(vq_tokens)
            test_text_tokens.append(text_tokens)
        if i % 200 == 0:
            np.save(os.path.join(OUT_DIR, "test_vq_tokens.npy"), np.concatenate(test_vq_tokens))
            np.save(os.path.join(OUT_DIR, "test_text_tokens.npy"), np.concatenate(test_text_tokens))
    test_vq_tokens = np.concatenate(test_vq_tokens)
    test_text_tokens = np.concatenate(test_text_tokens)
    np.save(os.path.join(OUT_DIR, "test_vq_tokens.npy"), test_vq_tokens)
    np.save(os.path.join(OUT_DIR, "test_text_tokens.npy"), test_text_tokens)

if __name__ == '__main__':
    main()